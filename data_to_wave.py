import os
import numpy as np
import pandas as pd
import scipy.io.wavfile

def main(directory_name):
    directory_path = "./result/" + directory_name
    df = pd.read_csv(directory_path + "/df.csv", index_col=0)
    os.makedirs("./wave/" + directory_name, exist_ok=True)

    u = np.load(directory_path + "/u.npy")
    mesh_file = df.iloc[0, 0]

    df = pd.read_csv(directory_path + "/df.csv", index_col=0)
    t_end = df.iloc[0, 1]
    samplerate = df.iloc[0, 2]
    t_detail = df.iloc[0, 3]

    cells_num = np.load("./parse_" + mesh_file.replace(".msh", "") + "/cells_num.npy")
    parse_directory = "./parse_" + mesh_file.replace(".msh", "") + "/"

    M = int(t_end * samplerate * t_detail)

    abs_max = np.amax(np.abs(u))
    target_cell_wave_data_normalized = u.T / abs_max

    for i in range(cells_num):
        scipy.io.wavfile.write("./wave/" + directory_name + "/" + str(i+1) + ".wav", samplerate, target_cell_wave_data_normalized[i, 0:M:t_detail])

    print("Save as wav format.")

if __name__ == "__main__":
    directory_name = "響板.msh_time:0.1_time_detail:100_2024-04-24_01:16:22"
    main(directory_name)