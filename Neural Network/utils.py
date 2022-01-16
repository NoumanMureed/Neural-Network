import math
import random


def random_matrix(rows, columns):
    matrix = []
    for _ in range(rows):
        row = [random.uniform(-1, 1) for _ in range(columns)]
        matrix.append(row)
    return matrix


def zero_matrix(rows, columns):
    matrix = []
    for _ in range(rows):
        row = [0 for _ in range(columns)]
        matrix.append(row)
    return matrix


def sigmoid(x, lambdaa):
    return 1 / (1 + math.exp(- lambdaa * x))


def root_mean_square_error(dataset_erros):

    sum_error1 = 0
    sum_error2 = 0
    data_sets_len = len(dataset_erros)
    for ek in dataset_erros:
        sum_error1 += math.pow(ek[0], 2)
        sum_error2 += math.pow(ek[1], 2)
    mse_1 = sum_error1 / data_sets_len
    mse_2 = sum_error2 / data_sets_len

    rmse1 = math.sqrt(mse_1)
    rmse2 = math.sqrt(mse_2)
    avg = (rmse1 + rmse2) / 2

    return avg


def normalize(valuee, minimum, maximum):
    return (valuee - minimum) / (maximum - minimum)


def denormalize(valuee, minimum, maximum):
    return (valuee * (maximum - minimum)) + minimum
