from asp import evaluation, get_diam, get_diam_indices, get_distance_matrix, get_euclidean_distance, merge_one_point, one_step_diam_descent, run_gradient, write_current_data
from dataset import four_n_plus_one, get_two_dim_Bn
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



data = [np.array([0.1043772485727064, -6.219163658130961e-05, -0.036990598090919855, 0.9938496273408105]), np.array([-0.4700736247234575, 0.8139520183197264, -0.33224698336966585, 0.07826136501717165]), np.array([-0.4700738077724748, -0.8139532958391443, -0.3322471984244794, 0.07824606429696979]), np.array([-0.00013774372708242352, -4.210407202265588e-07, 0.9969296766073409, 0.0783019854542087]), np.array([0.94280392617836, -5.644069974974151e-13, -0.33333141618208945, 0.0033051125113148306]), np.array([-0.4714037535053653, 0.8164952529168821, -0.33333278967825836, 0.0018040151217728155]), np.array([-0.47140375529563067, -0.8164952559948819, -0.33333279095523194, 0.0018019170416869307]), np.array([2.0658747978851887e-10, -2.4049148919972533e-14, 0.9999983793288768, 0.0018003720781986023]), np.array([0.942809013895621, -3.661502528981075e-10, -0.3333333228580278, -0.00024328911241710443]), np.array([-0.4714044478828691, 0.8164964547970915, -0.3333332819172411, -0.0005558653698189792]), np.array([-0.4714044478828171, -0.8164964547971366, -0.33333328191720435, -0.0005558653698189792]), np.array([-1.106606831907155e-13, 0.0, 0.9999998454644214, -0.0005559416636164467]), np.array([0.942744712986818, 7.268912600338509e-08, -0.33331091797226997, -0.011672107602292831]), np.array([-0.4713450402452529, 0.8164182045119677, -0.33330637814418196, -0.014213608481435918]), np.array([-0.4713449971454667, -0.8164180888440808, -0.3333063494360755, -0.014222352116138905]), np.array([1.411293284328889e-05, 1.6344080325822022e-08, 0.9998991901922306, -0.01419891734415976]), np.array([0.8486288970988483, 4.255493887678088e-06, -0.3000202879474392, -0.435679723892011]), np.array([-0.46245730572502963, 0.7748613114361447, -0.3109881920231823, -0.29834465445715097]), np.array([-0.4624467365442858, -0.774835268936937, -0.3109793726979206, -0.2984378522027195]), np.array([-0.015090377239742228, 2.793790237348218e-06, 0.9543585614989208, -0.2982817738325082])]


distance_matrix = get_distance_matrix(data)

min_dist = float('inf')
for i in range(len(data)):
    for j in range(i+1, len(data)):
        if distance_matrix[i, j] < min_dist:
            min_dist = distance_matrix[i, j]

print('min_dist: ', min_dist)

running = True




n_of_iteration = 100000
precision = 1e-7
output = "examine_3n_5_output.txt"
distance_matrix = get_distance_matrix(data)
diam = get_diam(distance_matrix)
diam_indices = get_diam_indices(distance_matrix, precision)
G = nx.from_dict_of_lists(diam_indices)

num_edge = G.number_of_edges()
print('num_edge: ', num_edge)

print('diam_indices: ', diam_indices)
loss = evaluation(data, distance_matrix, 1e-6)
print('loss: ', loss)

# write_current_data(output, 10, data, diam, diam_indices, precision)

run_gradient(data, 100000,  1e-6,output)
for i in range(n_of_iteration):
    # print('iteration: ', i)
    if running:
        # running, loss = one_step_gradient_descent(data, distance_matrix, 0.00001, 10e-4,  random.uniform(0, 0.00001))
        running = one_step_diam_descent(data, distance_matrix, precision + random.uniform(0, precision), 1e-8, 0)
        # running, loss = one_step_gradient_descent(data, distance_matrix, precision + random.uniform(0, precision), 1e-5, 0)
        # merge_one_point(data, distance_matrix, 0.0001)
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
        if loss <= 1e-8:
            running = False
            exit()
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


