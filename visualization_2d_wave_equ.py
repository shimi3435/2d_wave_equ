import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

class Visual:
    def __init__(self, directory):
        self.__directory = directory

    def prepare_visualization(self):

        with open("./result/" + self.__directory + "/out.txt", mode = "r") as f:
            solve_time = float(f.readline().rstrip())
            if f.readline().rstrip() == "True":
                allocate = True
            else:
                allocate = False

        df = pd.read_csv("./result/" + self.__directory + "/df.csv", index_col=0)
        self.__u = np.load("./result/" + self.__directory + "/u.npy")
        mesh_file = df.iloc[0, 0]
        parse_directory = "./parse_" + mesh_file.replace(".msh", "") + "/"

        self.__t_end = df.iloc[0, 1]
        samplerate = df.iloc[0, 2]
        t_detail = df.iloc[0, 3]
        self.__M = int(self.__t_end * samplerate * t_detail)
        self.__force_x = df.iloc[0, -4]
        self.__force_y = df.iloc[0, -3]

        cells_num = np.load("./parse_" + mesh_file.replace(".msh", "") + "/cells_num.npy")
        neighbor_tags = np.load("./parse_" + mesh_file.replace(".msh", "") + "/neighbor_tags.npy")
        neighbor_points = np.load("./parse_" + mesh_file.replace(".msh", "") + "/neighbor_points.npy")
        self.__physical_groups_num = np.load(parse_directory + "physical_groups_num.npy")
        self.__physical_groups_sizes = np.load(parse_directory + "physical_groups_sizes.npy")
        self.__triangle_centroid = np.load(parse_directory + "triangle_centroid.npy")

        bound_cond_points = []
        for i in range(cells_num):
            for j in range(3):
                if neighbor_tags[i,j] == -1:
                    bound_cond_points.append(neighbor_points[i, j])
        self.__bound_cond_points = np.asarray(bound_cond_points)

        bound_cond_len = len(self.__bound_cond_points)
        self.__bound_cond_np = np.zeros(bound_cond_len)

    def save_visualization(self):
        t_eval = np.linspace(0, self.__t_end, self.__M)
        dt = t_eval[1] - t_eval[0]

        os.makedirs("./visualization/" + self.__directory, exist_ok=True)

        fig_energy = plt.figure(figsize = (8, 8))
        ax_energy = fig_energy.add_subplot()
        ax_energy.set_xlabel("Time", size = 16)
        ax_energy.set_ylabel("Energy", size = 16)
        ax_energy.grid(True)
        ax_energy.tick_params(labelsize = 12)
        energy = np.load("./result/" + self.__directory + "/energy.npy")
        ax_energy.scatter(t_eval[:len(energy)], energy)
        fig_energy.savefig("./visualization/" + self.__directory + "/energy.png")
        print("Save as ../../visualization/" + self.__directory + "/energy.png")

        colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

        fig_cell_tags = plt.figure(figsize = (8, 8))
        ax_cell_tags = fig_cell_tags.add_subplot()
        ax_cell_tags.set_xlabel("x", size = 16)
        ax_cell_tags.set_ylabel("y", size = 16)
        start = 0
        end = 0
        for i in range(self.__physical_groups_num):
                    end = end + self.__physical_groups_sizes[i]
                    ax_cell_tags.scatter(self.__triangle_centroid[start:end, 0], self.__triangle_centroid[start:end, 1], c=colorlist[i % 8])
                    start = start + self.__physical_groups_sizes[i]
        ax_cell_tags.annotate("force center point", (self.__force_x, self.__force_y), size=16, color="black", arrowprops = dict(arrowstyle = "->", color = "black"), xytext = (0.9, 1.25))
        fig_cell_tags.savefig("./visualization/" + self.__directory + "/cell_tags.png")
        print("Save as ./visualization/" + self.__directory + "/cell_tags.png")

        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(projection="3d")
        gif_length = 1000 # Mを割り切れる数前提
        if self.__M % 1000 == 0:
            print("Animating result...")
            count = 0
            for i in range(0, self.__M, gif_length):
                t_start = count * gif_length * dt
                t_end = (count+1)*gif_length * dt - dt

                def update(j, fig_title):
                    time = dt * (count * gif_length + j)

                    if j != 0:
                        ax.cla()

                    ax.set_xlabel("x", size = 16)
                    ax.set_ylabel("y", size = 16)
                    ax.set_zlabel("u", size = 16)
                    ax.scatter(self.__bound_cond_points[:, 0], self.__bound_cond_points[:, 1], self.__bound_cond_np, c="black")

                    start = 0
                    end = 0
                    for i in range(self.__physical_groups_num):
                        end = end + self.__physical_groups_sizes[i]
                        ax.scatter(self.__triangle_centroid[start:end, 0], self.__triangle_centroid[start:end, 1], self.__u[count * gif_length + j, start:end], c=colorlist[i % 8])
                        start = start + self.__physical_groups_sizes[i]
                    ax.set_title(fig_title + "Time = " + str(format(time, '.6f')))

                ani = FuncAnimation(fig, update, fargs = ("",), interval=1, frames = gif_length)
                ani.save("./visualization/" + self.__directory + "/" + str(format(t_start, '.6f')) + "~" + str(format(t_end, '.6f')) + ".gif", writer="pillow")
                print("Save as " + str(format(t_start, '.6f')) + "~" + str(format(t_end, '.6f')) + ".gif")
                count = count + 1

            print("Done animating result")
        else:
            print("Cannot animating")

    @property
    def bound_cond_np(self):
        return self.__bound_cond_np

    @property
    def bound_cond_points(self):
        return self.__bound_cond_points

def main():
    directory = "響板.msh_time:0.1_time_detail:20_2024-05-14_22:00:24"
    v = Visual(directory)
    v.prepare_visualization()
    v.save_visualization()

if __name__ == "__main__":
    main()