import math
import random
from os.path import exists
import numpy as np
import miniball as mb
from scipy.optimize import linprog



def get_euclidean_distance(p:np.ndarray, q:np.ndarray) -> np.float64:
    """Computes Euclidean distance between two input vectors."""
    return np.linalg.norm(p - q)

def get_distance_matrix(data):
    """Computes the distance matrix using numpy's broadcasting feature."""
    data = np.array(data)
    data = data.reshape(data.shape[0], -1)
    distances = np.sqrt(np.sum((data[:, np.newaxis] - data[np.newaxis, :])**2, axis=2))
    return distances

def get_diam_indices(distance_matrix:np.ndarray, proximity: float) -> dict:
    """
    Returns indices of points that are within a certain proximity of the maximum diameter.

    Parameters:
    distance_matrix (np.ndarray): The pairwise distance matrix for points in the dataset.
    proximity (float): The proximity threshold.

    Returns:
    dict: A dictionary with keys as the point indices and values as the list of indices of points that are 
          within the proximity of the maximum diameter. Points with no nearby points are excluded.
    """
    n = distance_matrix.shape[0]
    max_diam = np.max(distance_matrix)

    # Using dictionary comprehension to create a dictionary of points and their nearby points
    indices_dict = {i: np.where(distance_matrix[i] + proximity >= max_diam)[0].tolist() for i in range(n)}

    # Using dictionary comprehension to remove keys where the values are empty lists
    indices_dict = {key: val for key, val in indices_dict.items() if val}

    return indices_dict


def update_data(new_point, index_to_change, data, distance_matrix):
    """Updates the distance matrix in-place when a point changes."""
    data[index_to_change] = new_point
    for i in range(len(data)):
        dist = get_euclidean_distance(data[i], new_point)
        distance_matrix[index_to_change, i] = dist
        distance_matrix[i, index_to_change] = dist
    return np.min(distance_matrix[index_to_change])


def one_step_diam_descent(data, distance_matrix, proximity, exit_threshold, min_step_length):
    """
    Performs one step of the diameter descent algorithm.

    Parameters:
    data (array): The dataset for the algorithm.
    distance_matrix (array): The matrix of distances between data points.
    proximity (float): The proximity threshold for considering points as a diameter.
    exit_threshold (float): The exit condition for the algorithm.
    min_step_length (float): The minimum step length for the descent.

    Returns:
    tuple: A boolean indicating whether the algorithm should continue, the maximum step length, the updated data, and the updated distance matrix.
    """
    # Get indices of diameters
    diam_indices = get_diam_indices(distance_matrix, proximity)

    # Special case: if there's only one point, we can't do anything
    if len(data) == 1:
        return False, 0, data, distance_matrix

    # Calculate differences for each point
    diff = {}
    for point in diam_indices.keys():
        # Special case: if this point is a diameter with all other points
        if len(diam_indices[point]) == len(data) - 1:
            diff[point] = [len(data) -1 , 0]
            continue

        # Find the second largest diameter involving this point
        second_diam = float('-inf')
        for i in range(len(data)):
            if i in diam_indices[point] or i == point:
                continue
            elif distance_matrix[point, i] > second_diam:
                second_diam = distance_matrix[point, i]

        diff[point] = [len(diam_indices[point]), distance_matrix[point, diam_indices[point][0]] - second_diam]

    # Order points by the number of diameters they're involved in, and the difference to the second largest diameter
    second_diam_order = [k for k, v in sorted(diff.items(), key=lambda item: (item[1][0], -item[1][1]))]

    # Initialize some values for the upcoming loop
    max_key = second_diam_order[0]
    max_unit_gradient = data[0].shape[0]
    max_step_length = 0.0

    # Iterate over the ordered points
    for key in second_diam_order:
        # If this point is a diameter with only one other point
        if len(diam_indices[key]) == 1:
            unit_gradient = (data[diam_indices[key][0]] - data[key]) / np.linalg.norm(data[diam_indices[key][0]] - data[key])
            step_length = armijo_rule_step_length(data, key, unit_gradient)
        else:
            tangent_vectors = get_tangent_vectors(key, diam_indices[key], data)
            center = np.zeros(tangent_vectors[0].shape[0])
            for vec in tangent_vectors:
                center += vec
            center_norm = np.linalg.norm(center)
            if center_norm == 0.:
                continue

            initial_vec = center / center_norm
            raw_gradient, min_dot_product = get_minmax_vec_miniball(tangent_vectors, initial_vec, data[key])
            gradient_norm = np.linalg.norm(raw_gradient)
            if gradient_norm == 0.0:
                continue

            unit_gradient = raw_gradient / gradient_norm

            if min_dot_product == 0. or np.linalg.norm(raw_gradient) <= 10e-6:
                continue
            step_length = armijo_rule_step_length(data, key, unit_gradient)

        # Update the maximum step length and gradient if necessary
        if step_length > max_step_length:
            max_key = key
            max_unit_gradient = unit_gradient
            max_step_length = step_length

        # If we've found a sufficiently large step, we can stop early
        if step_length > min_step_length:
            break

    # Calculate the gradient and the new point
    gradient = max_unit_gradient * max_step_length
    new_point = (data[max_key] + gradient) / np.linalg.norm(data[max_key] + gradient)

    # Update the data and distance matrix
    min_dist = update_data(new_point, max_key, data, distance_matrix)
    if min_dist < 0.001:
        data, distance_matrix = merge_one_point(data, distance_matrix,  0.0001)

    return True, max_step_length, data, distance_matrix



def get_exact_step_length(data, base_pt, target_pt):
    base_to_target = data[target_pt] - data[base_pt]
    step_length = 0.001
    for i in range(len(data)):
        if i == base_pt or i == target_pt:
            continue
        else:
            normal_vec = data[target_pt] - data[i]
            step_to_plane = -np.dot(normal_vec, data[base_pt])/np.dot(normal_vec, base_to_target)
            if step_to_plane < step_length and step_to_plane > 0:
                step_length = step_to_plane
    return step_length


# return unnormalized set of tangent vectors
def get_tangent_vectors(base_point_index, points_to_base_point, data):
    # raw_gradient = np.zeros(3)
    tangent_vectors = []
    base_point = data[base_point_index]
    for index in points_to_base_point:
        # print('index: ', index)
        raw_tangent_vector = data[index] - base_point
        tangent_vectors.append(raw_tangent_vector - np.dot(raw_tangent_vector, base_point) * base_point)
    return tangent_vectors


def get_minmax_vec_miniball(tangent_vectors, initial_vec, base_vec):
    """
    This function calculates the raw gradient and the minimum dot product based on the given tangent vectors.

    Parameters:
    tangent_vectors (list): The tangent vectors calculated for a given point.
    initial_vec (numpy.ndarray): The initial vector for the given point.
    base_vec (numpy.ndarray): The base vector used for normalization.

    Returns:
    raw_gradient (numpy.ndarray): The raw gradient calculated from the tangent vectors.
    min_dot_product (float): The minimum dot product.
    """
    if len(tangent_vectors) < 2:
        raw_gradient = initial_vec
        min_dot_product = np.dot(initial_vec, tangent_vectors[0])
        return raw_gradient, min_dot_product

    is_held = in_hull(tangent_vectors, np.zeros(initial_vec.shape[0]))
    if is_held:
        return np.zeros(initial_vec.shape[0]), 0.0
    else:
        g = initial_vec
        try:
            g, _ = mb.get_bounding_ball(np.stack(tangent_vectors, axis=0))
        except:
            print("an exception occurred")
        min_dot_product = np.inf
        for vec in tangent_vectors:
            curr_dot = np.dot(g, vec)
            if curr_dot < min_dot_product:
                min_dot_product = curr_dot
        return g, min_dot_product





def merge_one_point(data, distance_matrix, proximity):
    points_to_delete = set()
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if distance_matrix[i, j] <= proximity:
                points_to_delete.add(j)
                break
    data = get_reduced_set(data, points_to_delete)
    distance_matrix = get_distance_matrix(data)
    return data, distance_matrix


def get_reduced_set(data, points_to_delete):
    new_data = []
    for i in range(len(data)):
        if i not in points_to_delete:
            new_data.append(data[i])
    return new_data

def get_diam(distance_matrix:np.ndarray) -> np.float64:
    """[summary]

    Args:
        distance_matrix (np.ndarray): [description]

    Returns:
        np.float64: [description]
    """
    diam = 0.0
    n = distance_matrix.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if distance_matrix[i, j] > diam:
                diam = distance_matrix[i, j]
    return diam

def evaluation(data, distance_matrix, proximity_threshold):
    diam_indices = get_diam_indices(distance_matrix, proximity_threshold)
    loss = 0.
    for key in diam_indices.keys():
        tangent_vectors = get_tangent_vectors(key, diam_indices[key], data)
        center = np.zeros(data[0].shape[0])
        for vec in tangent_vectors:
            center += vec
        center_norm = np.linalg.norm(center)
        if center_norm == 0.:
            continue
        initial_vec = center/center_norm
        raw_gradient, min_dot_product = get_minmax_vec_miniball(tangent_vectors, initial_vec, data[key])
        loss += max(0, min_dot_product)
    if loss == 0:
        print('all points that reaches the diamter are been held with proximity threshold = ', proximity_threshold)
    return loss








def run_gradient(data, n_of_iteration, precision, output):
    if current_data_computed(output):
        return
    running = True
    distance_matrix = get_distance_matrix(data)
    for i in range(n_of_iteration):
        # print('iteration: ', i)
        if running:
            running, loss, data, distance_matrix = one_step_diam_descent(data, distance_matrix,  1e-1*precision + random.uniform(0, 1e-1*precision),  1e-3*precision, 1e-2*precision)

        else:
            break
        if i % 500 == 499:
            # merge_one_point(data, distance_matrix, 0.001)
            loss = evaluation(data, distance_matrix, precision)
            # print('iteration: ', i)
            diam = get_diam(distance_matrix)
            diam_indices = get_diam_indices(distance_matrix, precision)
            write_current_data(output, i, data, diam, diam_indices, precision)
            if loss <= 1e-6:
                running = False
    if running == False:
        diam_indices = get_diam_indices(distance_matrix, precision)
        final_data = []
        for index in diam_indices.keys():
            if diam_indices[index][0] > -1:
                final_data.append(data[index])
        final_distance_matrix = get_distance_matrix(final_data)
        loss = evaluation(final_data, final_distance_matrix, precision)
        # print('final_loss: ', loss)
        final_diam = get_diam(distance_matrix)
        final_diam_indices = get_diam_indices(final_distance_matrix, precision)
        write_current_data(output, i, final_data, final_diam, final_diam_indices, precision, True)
    return

def current_data_computed(output):
    is_computed = False
    if not exists(output):
        return False
    with open(output, "r") as myFile:
        for myLine in myFile:
            line_split = myLine.split()
            if len(line_split) == 0:
                continue
            initial_word = line_split[0]
            if initial_word == "success":
                is_computed=True
    return is_computed
    
                

def write_current_data(output,iteration, data, diam, diam_indices, precision, success=False):

    f = open(f'{output}', 'w+')
    f.truncate(0)
    if success:
        f.write(f'success with {10*precision}: \n')
        # print('precision: ', precision)
    f.write(f'This is iteration: {iteration}\n')
    f.write(f'data with {len(data)} points: \n')
    f.write('[')
    for i in range(len(data)):
        line = data[i]
        if len(line) == 4:
            f.write('np.array(['+ str(line[0])+', '+ str(line[1]) + ', ' + str(line[2]) + ', ' + str(line[3]) + '])')
        else:
            f.write('np.array(['+ str(line[0])+', '+ str(line[1]) + ',' + str(line[2]) + '])')
        if i != len(data) -1:
            f.write(', ')
    f.write(']\n')
    f.write(f'Euclidean diam: {diam}\n')
    s_diam = 2*math.asin(diam/2)
    f.write(f'Spherical diam: {s_diam}\n')
    # np.savetxt(f, distance_matrix)
    f.write(f'\n diam indices: {diam_indices}\n')
    return



def binary_search_step_length(data, base_point_index: int, unit_gradient_direction, step_length= 0.01) -> float:
    init_diam = get_largest_distance_to_data(data, data[base_point_index])
    init_point = data[base_point_index]
    l = 0
    r = step_length
    iteration = 0
    while l<=r and iteration <100:
        # print('here')
        mid = l + (r-l)/2
        new_point = (init_point + mid*unit_gradient_direction)/np.linalg.norm(init_point + mid*unit_gradient_direction)
        new_diam = get_largest_distance_to_data(data, new_point)
        if new_diam > init_diam:
            r = mid
        else:
            l = mid
        iteration += 1
    return l

def armijo_rule_step_length(data, base_point_index: int, unit_gradient_direction, step_length= 0.02) -> float:
    init_diam = get_largest_distance_to_data(data, data[base_point_index])
    init_point = data[base_point_index]
    r = step_length
    for i in range(20):
        # print('here')
        r = r/2
        new_point = (init_point + r*unit_gradient_direction)/np.linalg.norm(init_point + r*unit_gradient_direction)
        new_diam = get_largest_distance_to_data(data, new_point)
        if init_diam - new_diam > r*1e-3:
            break
    return r


def get_largest_distance_to_data(data, base_point)-> float:
    diam = float('-inf')
    for i in range(len(data)):
        diam = max(diam, get_euclidean_distance(data[i], base_point))
    return diam



'''
https://stackoverflow.com/a/43564754
'''

def in_hull(points, x):
    points = np.stack(points, axis=0)
    n_points = len(points)
    if n_points < 3:
        return False
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

