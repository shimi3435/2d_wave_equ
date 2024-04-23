import sys
import os
import shutil
import csv

import numpy as np
import jax
import jax.numpy as jnp

import gmsh

@jax.jit
def calc_triangle_area(vertex):
    v1 = jnp.array(vertex[0])
    v2 = jnp.array(vertex[1])
    v3 = jnp.array(vertex[2])
    return jnp.abs(jnp.cross(v2-v1, v3-v1)) / 2.0

@jax.jit
def calc_triangle_centroid(vertex):
    return jnp.mean(vertex, axis=0)

@jax.jit
def calc_line_center(points):
    return jnp.mean(points, axis=0)

@jax.jit
def calc_line_lengths(points):
    return jnp.sqrt((points[0, 0] - points[1, 0])**2 + (points[0, 1] - points[1, 1])**2)

@jax.jit
def calc_matrix_for_nabla(target_triangle_centoroid, target_neighbor_points):
    delta_xy = target_neighbor_points - target_triangle_centoroid
    return jnp.dot(jnp.linalg.inv(jnp.dot(delta_xy.T, delta_xy)), delta_xy.T)

@jax.jit
def calc_matrix_for_dd(target_triangle_centoroid, target_neighbor_points):
    delta_xy = target_neighbor_points - target_triangle_centoroid
    delta_x_plus_y = delta_xy[:,0] + delta_xy[:,1]
    delta_xy = jnp.hstack([delta_xy, delta_x_plus_y[:, jnp.newaxis]])
    return jnp.linalg.inv(delta_xy)

@jax.jit
def calc_matrix_for_dduddx(target_triangle_centoroid, target_neighbor_points):
    delta_x = target_neighbor_points[:, 0] - target_triangle_centoroid[0]
    return delta_x.T / jnp.dot(delta_x.T, delta_x)

@jax.jit
def calc_matrix_for_dduddy(target_triangle_centoroid, target_neighbor_points):
    delta_y = target_neighbor_points[:, 1] - target_triangle_centoroid[1]
    return delta_y.T / jnp.dot(delta_y.T, delta_y)

def main():
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " file")
        exit

    gmsh.initialize()

    #filename = sys.argv[1]
    filename = "響板.msh"

    gmsh.open(filename)

    #parse phycical groups num and size
    physical_groups_num = len(gmsh.model.getPhysicalGroups())
    physical_groups_sizes = np.empty(physical_groups_num, dtype=int)
    physical_groups_names = []
    for i, group in enumerate(gmsh.model.getPhysicalGroups()):
        physical_groups_sizes[i] = len(gmsh.model.mesh.getElements(2, int(gmsh.model.getEntitiesForPhysicalGroup(group[0], group[1])))[1][0])
        physical_groups_names.append(gmsh.model.getPhysicalName(group[0], group[1]))

    #parse triangle, calc vertices, centroid and area
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2)
    cells_num = 0
    cell_tags = np.empty(0)
    triangle_vertices = np.empty(0)
    triangle_vertices_tags = np.empty(0)
    triangle_centroid = np.empty(0)
    areas = np.empty(0)
    for i, elemType in enumerate(elemTypes):
        if elemType == 2:
            cell_tags = elemTags[i]
            cells_num = len(cell_tags)
            triangle_vertices = np.empty((cells_num, 3, 2), dtype="float32")
            triangle_vertices_tags = np.empty((cells_num, 3), dtype="float32")
            triangle_centroid = np.empty((cells_num, 2), dtype="float32")
            areas = np.empty(cells_num, dtype="float32")
            for j, cell_tag in enumerate(cell_tags):
                vertices_tags = elemNodeTags[i][j * 3: (j + 1) * 3]
                vertices = jnp.asarray([gmsh.model.mesh.getNode(x)[0][:2] for x in vertices_tags])
                triangle_vertices[j] = vertices
                triangle_vertices_tags[j] = vertices_tags
                triangle_centroid[j] = calc_triangle_centroid(vertices)
                areas[j] = calc_triangle_area(vertices)

    #clac neighbor
    gmsh.model.mesh.createEdges()
    elementType = gmsh.model.mesh.getElementType("triangle", 1)
    edgeNodes = gmsh.model.mesh.getElementEdgeNodes(elementType)
    edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(edgeNodes)
    elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)
    edges2Elements = {}
    for i in range(len(edgeTags)): # 3 edges per triangle
        if not edgeTags[i] in edges2Elements:
            edges2Elements[edgeTags[i]] = [elementTags[i // 3]]
        else:
            edges2Elements[edgeTags[i]].append(elementTags[i // 3])

    neighbor_tags = np.full((cells_num, 3), fill_value=-1, dtype=int)
    triangle_edges = jnp.asarray([gmsh.model.mesh.getEdges([triangle_vertices_tags[i, 0], triangle_vertices_tags[i, 1], triangle_vertices_tags[i, 1], triangle_vertices_tags[i, 2], triangle_vertices_tags[i, 2], triangle_vertices_tags[i, 0]])[0] for i in range(cells_num)])

    for i, mytag in enumerate(cell_tags):
        for j, edge_tag in enumerate(triangle_edges[i]):
            neighbor_triangle_tags = edges2Elements[int(edge_tag)]
            if len(neighbor_triangle_tags) > 1:
                neighbor_tags[i][j] = int(np.where(cell_tags == [tag for tag in neighbor_triangle_tags if tag != mytag][0])[0])

    neighbor_points = np.empty((cells_num, 3, 2), dtype="float32")
    triangle_edges_len = np.empty((cells_num, 3), dtype = "float32")

    for i, neighbors in enumerate(neighbor_tags):
        for j, neighbor_index in enumerate(neighbors):
            if neighbor_index == -1:
                edge = triangle_edges[i][j]
                edge_points_index = int(np.where(edgeTags == edge)[0])
                edge_points_tags = edgeNodes[2*edge_points_index], edgeNodes[2*edge_points_index + 1]
                edge_points = jnp.asarray([gmsh.model.mesh.getNode(point)[0][:2] for point in edge_points_tags])
                neighbor_points[i][j] = calc_line_center(edge_points)
                triangle_edges_len[i][j] = calc_line_lengths(edge_points)
            else:
                neighbor_points[i][j] = triangle_centroid[neighbor_index]

                edge = triangle_edges[i][j]
                edge_points_index = int(np.where(edgeTags == edge)[0][0]) #共有している辺は2個edgeがあるはず，その1個で長さを計算すればいいと考えている
                edge_points_tags = edgeNodes[2*edge_points_index], edgeNodes[2*edge_points_index + 1]
                edge_points = jnp.asarray([gmsh.model.mesh.getNode(point)[0][:2] for point in edge_points_tags])
                triangle_edges_len[i][j] = calc_line_lengths(edge_points)

    mat_ATA_inv_AT = jax.vmap(calc_matrix_for_nabla)(triangle_centroid, neighbor_points)

    def calc_c_to_c(triangle_centroid, neighbor_points):
        return jnp.sqrt((neighbor_points[:, 0] - triangle_centroid[0]) ** 2 + (neighbor_points[:, 1] - triangle_centroid[1]) ** 2)

    c_to_c = jax.vmap(calc_c_to_c)(triangle_centroid, neighbor_points)

    directory = "./parse_" + filename.replace(".msh", "") + "/"
    os.makedirs(directory, exist_ok=True)
    shutil.rmtree(directory)
    os.makedirs(directory)

    with open(directory + "physical_groups_names.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(physical_groups_names)

    np.save(directory + "physical_groups_num", physical_groups_num)
    np.save(directory + "physical_groups_sizes", physical_groups_sizes)
    np.save(directory + "cells_num", cells_num)
    np.save(directory + "cell_tags", cell_tags)
    np.save(directory + "areas", areas)
    np.save(directory + "neighbor_points", neighbor_points)
    np.save(directory + "neighbor_tags", neighbor_tags)

    np.save(directory + "triangle_centroid", triangle_centroid)
    np.save(directory + "triangle_vertices", triangle_vertices)

    np.save(directory + "mat_ATA_inv_AT", mat_ATA_inv_AT)

    np.save(directory + "c_to_c", c_to_c)
    np.save(directory + "triangle_edges", triangle_edges_len)

if __name__ == "__main__":
    main()
