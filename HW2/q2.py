# q2.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class KernelPerceptron:
    def __init__(self, degree=3, iterations=5):
        """
        Initializes the Kernel Perceptron.

        Parameters:
        - degree (int): Degree of the polynomial kernel.
        - iterations (int): Number of training iterations.
        """
        self.degree = degree
        self.iterations = iterations
        self.support_vectors = []
        self.alphas = []
        self.classes = None

    def polynomial_kernel(self, X, Y):
        """Computes the polynomial kernel between X and Y."""
        return (1 + np.dot(X, Y.T)) ** self.degree

    def fit(self, X, y):
        """
        Trains the Kernel Perceptron.

        Parameters:
        - X (np.ndarray): Training data of shape (n_samples, n_features).
        - y (np.ndarray): Training labels of shape (n_samples,).
        """
        self.classes = np.unique(y)
        for it in range(1, self.iterations + 1):
            mistakes = 0
            for xi, yi in zip(X, y):
                kernels = self.polynomial_kernel(np.array([xi]), np.array(self.support_vectors)) if self.support_vectors else np.array([])
                scores = {}
                for cls in self.classes:
                    cls_indices = np.where(np.array(self.classes) == cls)[0]
                    if len(self.alphas) > 0:
                        scores[cls] = np.sum(self.alphas * (self.classes[self.classes == cls] == cls) * kernels.flatten())
                    else:
                        scores[cls] = 0
                y_pred = max(scores, key=scores.get) if scores else None
                if y_pred != yi:
                    mistakes += 1
                    self.support_vectors.append(xi)
                    self.alphas.append(1)
            print(f"Iteration {it}/{self.iterations}, Mistakes: {mistakes}")
            if mistakes == 0:
                print("No mistakes in this iteration, stopping early.")
                break

    def predict(self, X):
        """
        Predicts labels for given data.

        Parameters:
        - X (np.ndarray): Data to predict of shape (n_samples, n_features).

        Returns:
        - np.ndarray: Predicted labels.
        """
        if not self.support_vectors:
            return np.array([None] * len(X))
        
        kernels = self.polynomial_kernel(X, np.array(self.support_vectors))
        scores = {}
        for cls in self.classes:
            cls_indices = np.where(np.array(self.classes) == cls)[0]
            scores[cls] = np.sum(self.alphas * (self.classes[self.classes == cls] == cls) * kernels, axis=1)
        predictions = np.argmax([scores[cls] for cls in self.classes], axis=0)
        return self.classes[predictions]


def plot_mistakes(mistakes_per_iteration, save_path='mistakes_plot.png'):
    """
    Plots the number of mistakes per iteration.

    Parameters:
    - mistakes_per_iteration (list): Number of mistakes in each iteration.
    - save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mistakes_per_iteration) + 1), mistakes_per_iteration, marker='o', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Mistakes')
    plt.title('Mistakes per Iteration in Kernel Perceptron')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def q2_main(best_degree, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Executes Part 2: Trains Kernel Perceptron with the best polynomial degree.

    Parameters:
    - best_degree (int): Optimal polynomial degree from Part 1(c).
    - X_train, y_train (np.ndarray): Training data and labels.
    - X_val, y_val (np.ndarray): Validation data and labels.
    - X_test, y_test (np.ndarray): Testing data and labels.

    Returns:
    - tuple: (train_acc, val_acc, test_acc, mistakes_per_iteration)
    """
    kp = KernelPerceptron(degree=best_degree, iterations=5)
    kp.fit(X_train, y_train)
    
    # Predict and evaluate
    y_train_pred = kp.predict(X_train)
    y_val_pred = kp.predict(X_val)
    y_test_pred = kp.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")
    
    # Note: Mistakes per iteration are not tracked in this concise version
    return train_acc, val_acc, test_acc, []


if __name__ == "__main__":
    """
    Executes Part 2 independently.
    Loads data, retrieves the best degree from Part 1(c), trains Kernel Perceptron, and evaluates performance.
    """
    import q1
    import q1c

    # Execute Part 1(a) and Part 1(b)
    C_best, accuracy_test, conf_matrix, X_train, X_val, X_test, y_train, y_val, y_test = q1.q1_main()

    # Execute Part 1(c)
    best_degree = q1c.q1c_main(C_best, X_train, y_train, X_val, y_val, X_test, y_test)

    # Execute Part 2
    train_acc, val_acc, test_acc, _ = q2_main(best_degree, X_train, y_train, X_val, y_val, X_test, y_test)
