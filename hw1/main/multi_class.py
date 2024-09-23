import numpy as np
import os

# Define the path to your data
data_dir = '/Users/Lerberber/Desktop/ML-CPT-570/hw1/data'

# Load data function for multi-class classification
def load_data(dataset_type='train', num_classes=10):
    
    labels_path = os.path.join(data_dir, f'{dataset_type}-labels-idx1-ubyte')
    images_path = os.path.join(data_dir, f'{dataset_type}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        y_data = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        x_data = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(y_data), 784)

    return x_data, y_data

# Weight-vector representation for multi-class classification
def get_weight_vector(x, y, num_classes=10):
    vector = np.zeros(len(x) * num_classes)
    for i in range(len(x)):
        vector[len(x) * y + i] = x[i]
    return vector

# Evaluation function for multi-class classification
def evaluate(weights, num_classes=10):
    
    x_test, y_test = load_data("t10k", num_classes)
    correct = 0
    total = len(x_test)
    
    for i in range(total):
        all_scores = [np.dot(weights, get_weight_vector(x_test[i], k, num_classes)) for k in range(num_classes)]
        y_pred = np.argmax(all_scores)
        if y_pred == y_test[i]:
            correct += 1

    return correct / total

# Multi-class Perceptron algorithm
def run_perceptron(iteration, num_classes=10, kind='train'):
    
    x_train, y_train = load_data(kind, num_classes)
    weight_vector = np.zeros(len(x_train[0]) * num_classes)
    mistake_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    
    for i in range(iteration):
        mistake = 0
        correct = 0
        for j in range(len(x_train)):
            all_scores = [np.dot(weight_vector, get_weight_vector(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)
            if y_pred != y_train[j]:
                mistake += 1
                weight_vector = update_weights_perceptron(weight_vector, x_train[j], y_train[j], y_pred, num_classes)
            else:
                correct += 1
        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))
        test_accuracy_list.append(evaluate(weight_vector, num_classes))

    return weight_vector, mistake_list, train_accuracy_list, test_accuracy_list

# Update weights for Perceptron
def update_weights_perceptron(weight_vector, xt, yt, y_pred, num_classes=10):
    return weight_vector + get_weight_vector(xt, yt, num_classes) - get_weight_vector(xt, y_pred, num_classes)

# Multi-class Passive-Aggressive algorithm
def run_passive_aggressive(iteration, num_classes=10, kind='train'):
    
    x_train, y_train = load_data(kind, num_classes)
    weight_vector = np.zeros(len(x_train[0]) * num_classes)
    mistake_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    
    for i in range(iteration):
        mistake = 0
        correct = 0
        for j in range(len(x_train)):
            all_scores = [np.dot(weight_vector, get_weight_vector(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)
            if y_pred != y_train[j]:
                mistake += 1
                weight_vector = update_weights_pa(weight_vector, x_train[j], y_train[j], y_pred, num_classes)
            else:
                correct += 1
        mistake_list.append(mistake)
        train_accuracy_list.append(correct / len(x_train))
        test_accuracy_list.append(evaluate(weight_vector, num_classes))

    return weight_vector, mistake_list, train_accuracy_list, test_accuracy_list

# Update weights for Passive-Aggressive algorithm
def update_weights_pa(weight_vector, xt, yt, y_pred, num_classes=10):
    learning_rate = (1 - (np.dot(weight_vector, get_weight_vector(xt, yt, num_classes)) - np.dot(weight_vector, get_weight_vector(xt, y_pred, num_classes)))) \
                    / np.linalg.norm(get_weight_vector(xt, yt, num_classes) - get_weight_vector(xt, y_pred, num_classes))
    return weight_vector + learning_rate * (get_weight_vector(xt, yt, num_classes) - get_weight_vector(xt, y_pred, num_classes))

# Incremental run for Perceptron and Passive-Aggressive algorithms
def increment_run(iteration, size, update_fn, num_classes=10):
    
    x_train, y_train = load_data("train", num_classes)
    weight_vector = np.zeros(len(x_train[0]) * num_classes)
    data_size = []
    test_accuracy_list = []
    
    for i in range(iteration):
        for j in range(len(x_train)):
            all_scores = [np.dot(weight_vector, get_weight_vector(x_train[j], k, num_classes)) for k in range(num_classes)]
            y_pred = np.argmax(all_scores)
            if y_pred != y_train[j]:
                weight_vector = update_fn(weight_vector, x_train[j], y_train[j], y_pred, num_classes)
            
            if (j + 1) % size == 0:
                test_accuracy_list.append(evaluate(weight_vector, num_classes))
                data_size.append((i + 1) * (j + 1))
    
    return weight_vector, data_size, test_accuracy_list
