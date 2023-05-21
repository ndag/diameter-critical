from asp import evaluation, get_diam, get_diam_indices, get_distance_matrix, get_euclidean_distance, merge_one_point, one_step_diam_descent, run_gradient, write_current_data
from dataset import four_n_plus_one, get_two_dim_Bn
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



data = [np.array([-0.34577747567218714, -0.600308432119284, -0.2459238288722711, 0.6779300804928625]), np.array([0.001015089302292866, 0.0003642818192259488, 0.7349821411417921, 0.6780856060226402]), np.array([0.9426695367202567, 2.6429579147226385e-06, -0.3332826694259882, 0.017228081521717238]), np.array([-0.47128209581504993, 0.8163614348076269, -0.333299940348209, 0.01892996795790745]), np.array([-0.47129201098503687, -0.8163366931302966, -0.3332959877630365, 0.019799706357269534]), np.array([1.9478320757831417e-05, 1.3066279796053227e-05, 0.9997991161230827, 0.02004312474066766]), np.array([0.934290882016138, -7.809997472030708e-05, -0.33024324714957937, -0.13431284150818484]), np.array([-0.4674360683133643, 0.8092223456384698, -0.330260158348753, -0.13263086053620227]), np.array([-0.46744645890524744, -0.8093593276807611, -0.33025662867073297, -0.13176435774568565]), np.array([-0.00020954789427112707, -6.75881606621639e-05, 0.9913107116518476, -0.13154096124716508]), np.array([-0.22544543017677363, -0.38872938130090523, 0.31788196605148394, -0.8348741712288554])]


distance_matrix = get_distance_matrix(data)

min_dist = float('inf')
for i in range(len(data)):
    for j in range(i+1, len(data)):
        if distance_matrix[i, j] < min_dist:
            min_dist = distance_matrix[i, j]

print('min_dist: ', min_dist)

running = True




n_of_iteration = 100000
precision = 1e-5
output = "examine_t5_output.txt"
distance_matrix = get_distance_matrix(data)
diam = get_diam(distance_matrix)
diam_indices = get_diam_indices(distance_matrix, precision)
G = nx.from_dict_of_lists(diam_indices)

num_edge = G.number_of_edges()
print('num_edge: ', num_edge)

print('diam_indices: ', diam_indices)
loss = evaluation(data, distance_matrix, 1e-5)
print('loss: ', loss)

# write_current_data(output, 10, data, diam, diam_indices, precision)

run_gradient(data, 100000,  1e-5,output)
for i in range(n_of_iteration):
    # print('iteration: ', i)
    if running:
        # running, loss = one_step_gradient_descent(data, distance_matrix, 0.00001, 10e-4,  random.uniform(0, 0.00001))
        running = one_step_diam_descent(data, distance_matrix, precision + random.uniform(0, precision), 1e-7, 0)
        # running, loss = one_step_gradient_descent(data, distance_matrix, precision + random.uniform(0, precision), 1e-5, 0)
        merge_one_point(data, distance_matrix, 0.001)
        # diam_tmp = get_diam(distance_matrix)
        # if len(data) <10 or diam_tmp<1.731:
        #     running = False

        distance_matrix = get_distance_matrix(data)

        # print('running: ', running)
        # print('loss: ', loss)
        # print('number of points= ', len(data))
        # print(data)
    else:
        break
    if i % 500 == 499:
        loss = evaluation(data, distance_matrix, precision*10)
        print('iteration: ', i)
        # print('Eva:loss: ', loss)
        # print('Eva:data: ', data)
        # print('Eva:distance matrix', distance_matrix)
        # print('eva:diam indices: ', get_diam_indices(distance_matrix))
        diam = get_diam(distance_matrix)
        diam_indices = get_diam_indices(distance_matrix, 10*precision)
        write_current_data(output, i, data, diam, diam_indices, precision)
        if loss <= 1e-4:
            running = False
if running == False and len(data)>3:
    final_data = []
    for index in diam_indices.keys():
        if diam_indices[index][0] >2:
            final_data.append(data[index])
    final_distance_matrix = get_distance_matrix(final_data)
    loss = evaluation(final_data, final_distance_matrix, precision*10)
    final_diam = get_diam(distance_matrix)
    final_diam_indices = get_diam_indices(final_distance_matrix, precision*10)
    write_current_data(output, i, final_data, final_diam, final_diam_indices, precision, True)


