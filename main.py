import numpy as np
import math
import random
import copy

ALPHA = 0.2
E = 0.01


def multiplication(matrix_1, matrix_2):
    matrix_1 = copy.deepcopy(matrix_1)
    matrix_2 = copy.deepcopy(matrix_2)
    transposed_matrix_2 = list(zip(*matrix_2))

    return [[sum(el1 * el2 for el1, el2 in zip(row_1, col_2)) for col_2 in transposed_matrix_2] for row_1 in matrix_1]


def ELU(x_input):
    if x_input >= 0:
        return x_input
    if x_input < 0:
        result = ALPHA * (math.exp(x_input) - 1)
        return result


def ELU_dif(x_input):
    if x_input > 0:
        return 1
    else:
        return ELU(x_input) + ALPHA


def sigma_func(x_input):
    return 1/(1 + math.exp((-1) * x_input))


def sigma_func_def(x_input):
    return sigma_func(x_input) * (1 - sigma_func(x_input))


def factorial(x_input):
    result = 1
    for i in range(1, x_input + 1):
        result *= i
    return result


def create_window(sequence, rows, columns):
    window = []
    for row in range(rows):
        new_row = []
        for col in range(columns):
            new_row.append(sequence[row + col])
        window.append(new_row)
    return window


def predict_next(W1, W2, line):
    input_layer = np.array(line + [0])
    hidden_layer = np.matmul(input_layer, W1)
    output_layer = np.matmul(hidden_layer, W2)
    result = math.fabs(output_layer[0])
    return result


choice = input("Выберите режим работы: 1 - обучение; !1 - предсказание: ")
if choice == '1':
    X_choice = input("Выберите последовательность: \n1 - ряд Фиббоначи\n"
                     "2 - факториальная функция\n3 - периодическая функция\n4 - степенная функция\n\n")
    X_count = int(input("Выберите количество элементов в выбранной последовательности: "))
    X = []
    if X_choice == '1':
        row_type = "fib"
        for index_of_primary_fill in range(X_count):
            X.append(1)
        for index_of_fibonacci in range(2, X_count):
            X[index_of_fibonacci] = X[index_of_fibonacci - 1] + X[index_of_fibonacci - 2]
        print(X)
    if X_choice == '2':
        row_type = "fac"
        for index_of_primary_fill in range(X_count):
            X.append(1)
        for index_of_factorial in range(X_count):
            X[index_of_factorial] = factorial(index_of_factorial + 1)
        print(X)
    if X_choice == '3':
        row_type = "loop"
        for index_of_primary_fill in range(X_count):
            if index_of_primary_fill % 2 == 0:
                X.append(1)
            else:
                X.append(0)
        print(X)
    if X_choice == '4':
        row_type = "pow"
        x_pow = int(input("Введите степень для степенной функции: "))
        for index_of_pow_func in range(X_count):
            X.append(index_of_pow_func ** x_pow)
        print(X)

    E = float(input("Введите ошибку: "))
    ALPHA = float(input("Введите коэффициент обучения: "))
    L = int(input("Введите количество столбцов в матрице обучения: "))
    # p = int(input("Введите количество строк в матрице обучения: "))
    limit_of_iterations = int(input("Введите количество шагов обучения, которые может пройти сеть: "))

    # L = 4
    p = len(X) - L - 1
    # limit_of_iterations = 1000
    number_of_neurons_on_hidden_layer = 5

    learning_window = create_window(X, p, L)
    contexts_list = []
    for adding_contexts in range(len(learning_window)):
        new_context = [0]
        contexts_list.append(new_context)

    W1 = []
    W2 = []

    # ==================== ИНИЦИАЛИЗАЦИЯ ВЕСОВЫХ МАТРИЦ

    for init_w1_index_row in range(L + 1):
        new_row = []
        for init_w1_index_col in range(number_of_neurons_on_hidden_layer):
            weight = np.random.randn() / 10
            new_row.append(weight)
        W1.append(new_row)

    for init_w2_index_row in range(number_of_neurons_on_hidden_layer):
        weight = np.random.randn() / 10
        W2.append([weight])

    # ==================== ОБУЧЕНИЕ

    sum_err = E + 1
    current_iteration = 0
    while sum_err > E and current_iteration < limit_of_iterations:
        current_iteration += 1
        for sample_index in range(len(learning_window) - 1):
            input_layer = np.array(learning_window[sample_index] + contexts_list[sample_index])
            hidden_layer = np.matmul(input_layer, np.array(W1))
            for activating_index in range(len(hidden_layer)):
                hidden_layer[activating_index] = sigma_func(hidden_layer[activating_index])
            output_layer = np.matmul(hidden_layer, np.array(W2))
            output_layer[0] = sigma_func(output_layer[0])

            contexts_list[sample_index][0] = output_layer[0]

            intended_outcome = learning_window[sample_index + 1][L - 1]
            err = output_layer[0] - intended_outcome

            # ============ КОРРЕКТИРОВКА МАТРИЦ

            XT_W2T = np.matmul(np.array([input_layer]).T, np.array(W2).T)
            inactive_hidden_layer = np.matmul([input_layer], W1)
            deactivated_hidden_layer = []
            for deactivating_index in range(len(inactive_hidden_layer[0])):
                deactivated_hidden_layer.append(sigma_func_def(inactive_hidden_layer[0][deactivating_index]))
            Xt_W2T_DHL = np.array(multiplication(list(XT_W2T), [deactivated_hidden_layer]))

            for multiplying_index in range(len(Xt_W2T_DHL)):
                Xt_W2T_DHL[multiplying_index][0] *= ALPHA
                Xt_W2T_DHL[multiplying_index][0] *= err

            W1 = np.array(W1) - Xt_W2T_DHL

            deactivated_output_layer = [0]
            H_W2 = np.matmul(hidden_layer, W2)
            deactivated_output_layer[0] = sigma_func_def(H_W2[0])
            HL_DOL = np.matmul(np.array([hidden_layer]).T, np.array([deactivated_output_layer]))

            for multiplying_index in range(len(HL_DOL)):
                HL_DOL[multiplying_index] *= ALPHA
                HL_DOL[multiplying_index] *= err

            W2 = W2 - HL_DOL

        sum_err = 0
        for sample_index in range(len(learning_window) - 1):
            input_layer = np.array(learning_window[sample_index] + contexts_list[sample_index])
            hidden_layer = np.matmul(input_layer, np.array(W1))
            output_layer = np.matmul(hidden_layer, np.array(W2))

            contexts_list[sample_index][0] = output_layer[0]

            intended_outcome = learning_window[sample_index + 1][L - 1]
            err = output_layer[0] - intended_outcome
            sum_err += err ** 2
        print("Итерация ", current_iteration, " ", sum_err)

    print("\n", str(predict_next(W1, W2, learning_window[-1])))

    with open("wm1", 'w') as weight_matrix_file:
        np.save("wm1", W1)
    with open("wm2", 'w') as weight_matrix_file:
        np.save("wm2", W2)

    # ====================================     всё остальное     =======================================

else:
    W1 = np.array([])
    W2 = np.array([])
    with open("wm1.npy", "r") as weight_matrix_file:
        W1 = np.load("wm1.npy")
    with open("wm2.npy", "r") as weight_matrix_file:
        W2 = np.load("wm2.npy")

    X_choice = input("Выберите последовательность: \n1 - ряд Фиббоначи\n2 - Факториальная функция\n\n")
    X_count = int(input("Выберите количество элементов в выбранной последовательности: "))
    X = []
    row_type = ""
    if X_choice == '1':
        row_type = "fib"
        for index_of_primary_fill in range(X_count):
            X.append(1)
        for index_of_fibonacci in range(2, X_count):
            X[index_of_fibonacci] = X[index_of_fibonacci - 1] + X[index_of_fibonacci - 2]
        print(X)
    if X_choice == '2':
        row_type = "fac"
        for index_of_primary_fill in range(X_count):
            X.append(1)
        for index_of_factorial in range(X_count):
            X[index_of_factorial] = factorial(index_of_factorial + 1)
        print(X)

    learning_window = create_window(X, 7, 4)
    print("\n", str(predict_next(W1, W2, X[-5:-1])))
