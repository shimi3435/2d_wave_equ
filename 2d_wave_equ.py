import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import time
import datetime
import shutil

def run_simulation(df):
    ### パラメータの読み込み ###
    mesh_file = df.iloc[0, 0]
    parse_directory = "parse_" + mesh_file.replace(".msh", "") + "/"

    physical_groups_names = []
    with open(parse_directory + "/physical_groups_names.csv", "r") as f:
        physical_groups_names = f.readline().rstrip().split(",")

    physical_groups_num = np.load(parse_directory + "physical_groups_num.npy")
    physical_groups_sizes = np.load(parse_directory + "physical_groups_sizes.npy")
    mat_ATA_inv_AT = np.load(parse_directory + "mat_ATA_inv_AT.npy")
    cells_num = np.load(parse_directory + "cells_num.npy")
    cell_tags = np.load(parse_directory + "cell_tags.npy")
    areas = np.load(parse_directory + "areas.npy")
    neighbor_tags = np.load(parse_directory + "neighbor_tags.npy")

    triangle_vertices = np.load(parse_directory + "triangle_vertices.npy")
    triangle_centroid = np.load(parse_directory + "triangle_centroid.npy")

    c_to_c = np.load(parse_directory + "c_to_c.npy")
    triangle_edges = np.load(parse_directory + "triangle_edges.npy")

    coefficient_param = np.empty((physical_groups_num, 8), dtype="float32")
    coefficient = np.empty((8, cells_num), dtype="float32")

    t_end = df.iloc[0, 1]
    samplerate = df.iloc[0, 2]
    t_detail = df.iloc[0, 3]
    for i in range(physical_groups_num):
        for j in range(6):
            coefficient_param[i][j] = df.iloc[0, 4 + 6 * i + j]
    resist_c = df.iloc[0, -6]
    force_c = df.iloc[0, -5]
    force_x = df.iloc[0, -4]
    force_y = df.iloc[0, -3]
    force_r = df.iloc[0, -2]
    force_file_path = df.iloc[0, -1]

    for i in range(6):
        start = 0
        end = 0
        for j in range(physical_groups_num):
            end = end + physical_groups_sizes[j]
            coefficient[i, start:end] = coefficient_param[j, i]
            start = end

    M = int(t_end * samplerate * t_detail)
    t_eval = np.linspace(0.0, t_end, M, dtype="float32")
    dt = t_eval[1] - t_eval[0]

    ###外力挿入点の決定###
    force_p = jnp.asarray([force_x, force_y])

    @jax.jit
    def is_point_inside_triangle(triangle):
        v1 = triangle[0]
        v2 = triangle[1]
        v3 = triangle[2]

        u = v1 - force_p
        v = v2 - force_p
        w = v3 - force_p

        uv = jnp.cross(u, v)
        vw = jnp.cross(v, w)
        wu = jnp.cross(w, u)

        uv_dot_vw = jnp.dot(uv, vw)
        wu_dot_vw = jnp.dot(wu, vw)

        return jnp.where(uv_dot_vw >= 0, 1, 0) * jnp.where(wu_dot_vw >= 0, 1, 0)

    @jax.jit
    def is_centoroid_inside_circle(centroid):
        distance = jnp.sqrt((force_x - centroid[0])**2 + (force_y - centroid[1])**2)
        return jnp.where(distance < force_r, 1.0, 0.0)

    force_point_cercle = jax.vmap(is_centoroid_inside_circle)(triangle_centroid)
    force_point = jax.vmap(is_point_inside_triangle)(triangle_vertices)
    force_point_cercle = force_point_cercle.at[jnp.where(force_point == 1)].set(1)
    force_point_mask = force_point_cercle

    force_points = jnp.where(force_point_mask == 1)
    print("force_points: " + str(force_points))

    force_areas = np.empty(len(force_points[0]), dtype="float32")
    for i, index in enumerate(force_points[0]):
        force_areas[i] = areas[index]

    force_areas_normalized = force_areas / force_areas.sum()
    for i, index in enumerate(force_points[0]):
        force_point_mask = force_point_mask.at[index].set(force_areas_normalized[i])

    force_power = np.zeros(M, dtype="float32")
    force_data = np.loadtxt(force_file_path)

    sol_energy = np.empty(M, dtype="float32")

    ###変位と速度の配列確保 できなかったら終了###
    try:
        sol = np.empty((M, 2, cells_num+1), dtype="float32")
        allocate = True
    except:
        allocate = False

    for i in range(len(force_data)):
        if(t_detail * i < len(force_data)):
            for j in range(t_detail):
                force_power[t_detail * i + j] = force_data[i]

    ###エネルギー（ハミルトニアン）の計算関数###
    @jax.jit
    def calc_energy(u, v):
        dudx_dudy = jax.vmap(lambda x,y,z : jnp.dot(x, y - z))(mat_ATA_inv_AT, u[neighbor_tags], u[:cells_num])
        energy = jnp.sum((coefficient[0] / 2.0 * v[:cells_num]**2 + coefficient[1] / 2.0 * dudx_dudy[:,0] ** 2 + coefficient[2]/ 2.0 * dudx_dudy[:,1] ** 2) * areas[:cells_num])
        return energy

    #dudtの右辺
    @jax.jit
    def rhs_u(u, v):
        return coefficient[0] * v[:cells_num]

    #dvdtの右辺
    @jax.jit
    def rhs_v(u, v, f):
        return coefficient[1] / areas * ((u[neighbor_tags[:, 0]] - u[:cells_num]) / c_to_c[:, 0] * triangle_edges[:, 0] + (u[neighbor_tags[:, 1]] - u[:cells_num]) / c_to_c[:, 1] * triangle_edges[:, 1] + (u[neighbor_tags[:, 2]] - u[:cells_num]) / c_to_c[:, 2] * triangle_edges[:, 2]) - resist_c * v[:cells_num] + force_c * f

    dt_now = datetime.datetime.now()
    output_directry = str(mesh_file) + "_time:" + str(t_end) + "_time_detail:" + str(t_detail) + "_" + dt_now.strftime("%Y-%m-%d_%H:%M:%S")

    os.makedirs("./result/" + output_directry + "/", exist_ok=True)
    shutil.rmtree("./result/" + output_directry + "/")
    os.makedirs("./result/" + output_directry + "/")

    if allocate:
        u0 = np.zeros(cells_num+1, dtype="float32")
        v0 = np.zeros(cells_num+1, dtype="float32")
        sol[0,0] = u0
        sol[0,1] = v0

        jax.device_put(areas)
        jax.device_put(coefficient)
        jax.device_put(force_c)
        jax.device_put(mat_ATA_inv_AT)
        jax.device_put(neighbor_tags)
        jax.device_put(cells_num)
        jax.device_put(force_power)
        jax.device_put(force_point_mask)

        print("Now solving...")
        solve_time_start = time.perf_counter()
        for n in range(M-1):
            tu = sol[n,0,:]
            tv = sol[n,1,:]
            energy = calc_energy(tu, tv)
            sol_energy[n] = energy
            nv = np.asarray(tv + dt * jnp.concatenate([rhs_v(tu, tv, force_power[n] * force_point_mask), jnp.asarray([0])]))
            nu = np.asarray(tu + dt * jnp.concatenate([rhs_u(tu, nv), jnp.asarray([0])])) #シンプレクティックオイラー法
            #nu = np.asarray(tu + dt * jnp.concatenate([rhs_u(tu, tv), jnp.asarray([0])])) #陽的オイラー法
            sol[n+1,0,:] = nu
            sol[n+1,1,:] = nv
        energy = calc_energy(sol[-1,0,:], sol[-1,1,:])
        sol_energy[M-1] = energy
        solve_time_end = time.perf_counter()
        solve_time = solve_time_end - solve_time_start
        print("solve time = " + str(solve_time))

        df.to_csv("./result/" + output_directry + "/df.csv")
        print("Save as ./result/" + output_directry + "/df.csv")

        with open("./result/" + output_directry + "/out.txt", "w") as f:
            f.write(str(solve_time))
            f.write("\n")
            f.write(str(allocate))
        print("Save as ./result/" + output_directry + "/out.txt")

        np.save("./result/" + output_directry + "/energy", sol_energy)
        print("Save as ./result/" + output_directry + "/energy.npy")

        np.savetxt("./result/" + output_directry + "/energy.nptxt", sol_energy)

        np.save("./result/" + output_directry + "/u", sol[:,0,:])
        print("Save as ./result/" + output_directry + "/u.npy")

        np.save("./result/" + output_directry + "/v", sol[:,1,:])
        print("Save as ./result/" + output_directry + "/v.npy")

    else:
        print("Cannot allocate.")

def main():
    df = pd.read_csv("./params.csv", index_col=0)
    run_simulation(df)

if __name__ == "__main__":
    main()