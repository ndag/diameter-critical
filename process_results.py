import ast
import os
import networkx as nx
import asp
import dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms import coloring
import itertools



def count_4cycles(G):
    cycle4 = nx.cycle_graph(4)
    count = 0
    combinations_of_4 =itertools.combinations(G.nodes(), 4)
    for nodes in combinations_of_4:
        subgraph = G.subgraph(nodes)
        if nx.is_isomorphic(subgraph, cycle4):
            count += 1
    return count




def subset_pairs_with_common_elements(S):
    count = 0
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            common_elements = len(set(S[i]).intersection(set(S[j])))
            if common_elements >= 3:
                count += 1
    return count


def count_connected_4_node_cliques(G):
    count = 0
    combinations_of_4 = itertools.combinations(G.nodes(), 4)
    for comb in combinations_of_4:
        subgraph = G.subgraph(comb)
        if nx.is_connected(subgraph) and nx.number_of_edges(subgraph) == 6:
            count += 1
    return count


def is_pyramid(poly):
    return poly.n_pts == poly.mFace + 1



def extract_data_from_file(file_name: str) -> dict:
    """
    Extract data from a file and return it as a dictionary
    """
    data = {}
    with open(file_name, "r") as f:
        for line in f:
            line_split = line.split()
            if not line_split:
                continue
            initial_word = line_split[0]
            if initial_word == "data":
                data["n_pts"] = int(line_split[2])
            elif initial_word == "Euclidean":
                data["e_diam"] = float("{:.5f}".format(float(line_split[2])))
            elif initial_word[0] == "[":
                data["points"] = eval(line)
            elif initial_word == "Spherical":
                data["s_diam"] = float("{:.5f}".format(float(line_split[2])))
            elif initial_word == "diam":
                data["diam_indices"] = ast.literal_eval(''.join(line_split[2:]))
    return data

def create_polygon_from_data(data: dict) -> dataset.ASDpolygon:
    """
    Create an ASDpolygon object from extracted data
    """
    mFace = max([len(data["diam_indices"][key]) for key in data["diam_indices"].keys()])
    n_mFace = sum([1 for key in data["diam_indices"].keys() if len(data["diam_indices"][key]) == mFace])
    G = nx.from_dict_of_lists(data["diam_indices"])
    n_tri = int(sum(nx.triangles(G).values()) / 3)
    n_tetrahedra = count_connected_4_node_cliques(G)
    n_ridge = count_4cycles(G)
    if len(data["points"][0]) ==4:
        n_edge = subset_pairs_with_common_elements(data["diam_indices"])
        return dataset.ASDpolygon(data["points"], mFace, n_mFace, n_tri, data["e_diam"], data["s_diam"], G, n_edge, n_tetrahedra=n_tetrahedra, n_ridge=n_ridge)
    return dataset.ASDpolygon(data["points"], mFace, n_mFace, n_tri, data["e_diam"], data["s_diam"], G)

def extract_polygons_from_directory(directory: str, is_no_more_than_ten=False) -> dict:
    """
    Extract polygons from a directory that meet certain conditions and return them in a dictionary
    """
    result = {}
    for filename in os.listdir(directory):
        if os.stat(filename).st_size == 0:
            continue
        with open(filename, "r") as f:
            line_split = f.readline().split()
            if line_split[0] != "success":
                continue
        data = extract_data_from_file(filename)
        polygon = create_polygon_from_data(data)
        # polygon.details()
        if polygon.n_pts >= 4:
            if polygon.n_pts < 11 or not is_no_more_than_ten:
                if polygon in result.keys():
                    result[polygon].append(filename)
                else:
                    result[polygon] = [filename]
    return result




os.chdir('./T5_results')
result_3 = extract_polygons_from_directory("./")



chromatic_count = 0
three_n_five_count = 0
parity_count = 0
three_n__minus_five_count = 0



three_n__minus_five_table = []
for polygon in result_3.keys():

    graph_dict = dict(polygon.diam_graph.adj)
    n_edge= subset_pairs_with_common_elements(graph_dict)
    total = polygon.n_pts + subset_pairs_with_common_elements(graph_dict)
    three_n__minus_five_value = 3* polygon.n_pts - n_edge
    if three_n__minus_five_value < 5: 
        print('three_n__minus_five_value: ', three_n__minus_five_value)  
        three_n__minus_five_table.append({'3f0-f1': three_n__minus_five_value, 'n_pts': polygon.n_pts, 'n_edge':polygon.n_edge, 'mFace': polygon.mFace,'n_mFace': polygon.n_mFace, 'n_tri': polygon.n_tri, 'n_tetra':polygon.n_tetrahedra, 'ridge':polygon.n_ridge, 'diam':polygon.s_diam, 'origin':result_3[polygon]})
        polygon.details()
        three_n__minus_five_count += 1

        
    if total %2 ==1:
        parity_count +=1 


    # Compute a proper vertex coloring using the greedy algorithm
    colors = coloring.greedy_color(polygon.diam_graph)

    # The chromatic number is the maximum number of colors used
    chromatic_number = max(colors.values()) + 1
    if chromatic_number > 5:
        # print('polygon.data: ', polygon.data)
        # print('chromatic_number: ', chromatic_number)
        polygon.details()
        print('borsuk counter example')
        print("#############################################")
        chromatic_count+= 1



print('three_n__minus_five_count: ', three_n__minus_five_count)
print('parity_count: ', parity_count)
print('chromatic_count: ', chromatic_count)

for item in three_n__minus_five_table:
    print(item)
    print('#############################################')


