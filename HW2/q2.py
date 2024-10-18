# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import fashion_mnist
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_and_preprocess_data(subset_size=10000):
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Select a subset of the data for training to manage computational load
    if subset_size < X_train_full.shape[0]:
        X_train_full, _, y_train_full, _ = train_test_split(
            X_train_full, y_train_full, train_size=subset_size, random_state=42, stratify=y_train_full
        )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    val_relative_size = 0.2 / (1 - 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=42, stratify=y_train_val
    )

    # Flatten images and scale
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


class KernelPerceptron:
    def __init__(self, kernel='poly', degree=3, iterations=5):
        self.kernel = kernel
        self.degree = degree
        self.iterations = iterations
        self.classes = None
        # For each class, store a list of (alpha, support_vector) tuples
        self.support_vectors = {cls: [] for cls in range(10)}

    def polynomial_kernel(self, x, y):
        return (1 + np.dot(x, y)) ** self.degree

    def fit(self, X, y):
        self.classes = np.unique(y)
        mistakes_per_iteration = []

        for it in range(self.iterations):
            mistakes = 0
            for idx in range(X.shape[0]):
                x_i = X[idx]
                y_i = y[idx]

                scores = {}
                for cls in self.classes:
                    score = 0
                    for alpha, sv in self.support_vectors[cls]:
                        score += alpha * self.polynomial_kernel(x_i, sv)
                    scores[cls] = score

                y_pred = max(scores, key=scores.get)

                if y_pred != y_i:
                    mistakes += 1
                    # Update support vectors
                    self.support_vectors[y_i].append((1, x_i))
                    self.support_vectors[y_pred].append((-1, x_i))
            mistakes_per_iteration.append(mistakes)
            print(f"Iteration {it + 1}/{self.iterations}, Mistakes: {mistakes}")
            if mistakes == 0:
                print("No mistakes in this iteration, stopping early.")
                break
        return mistakes_per_iteration

    def predict(self, X):
        predictions = []
        for idx, x in enumerate(X):
            if idx % 1000 == 0 and idx > 0:
                print(f"Predicting sample {idx}/{X.shape[0]}")
            scores = {}
            for cls in self.classes:
                score = 0
                for alpha, sv in self.support_vectors[cls]:
                    score += alpha * self.polynomial_kernel(x, sv)
                scores[cls] = score
            y_pred = max(scores, key=scores.get)
            predictions.append(y_pred)
        return np.array(predictions)


def plot_mistakes(mistakes_per_iteration, save_path='.'):
    plt.figure(figsize=(9, 7))
    plt.plot(range(1, len(mistakes_per_iteration) + 1), mistakes_per_iteration, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Mistakes')
    plt.title('Number of Mistakes vs Training Iterations')

    # Check if the directory exists, if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the figure
    plt.savefig(os.path.join(save_path, 'mistakes_plot.png'))

    # Show the plot
    plt.show()


def q2_main():
    # Adjust subset_size based on your computational resources
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(subset_size=10000)

    best_degree = None
    best_val_acc = 0
    best_model = None
    best_mistakes = []

    # Testing degrees 2, 3, and 4
    for degree in [2, 3, 4]:
        print(f"\nTraining Kernel Perceptron with polynomial degree {degree}")
        kp = KernelPerceptron(kernel='poly', degree=degree, iterations=5)
        mistakes_per_iteration = kp.fit(X_train, y_train)
        
        print("Predicting on validation set...")
        y_val_pred = kp.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy for degree {degree}: {val_acc * 100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_degree = degree
            best_model = kp
            best_mistakes = mistakes_per_iteration

    # Plot the number of mistakes per iteration for the best degree
    plot_mistakes(best_mistakes)

    # Predictions and accuracy calculation for the best model
    print("\nPredicting on training set...")
    y_train_pred = best_model.predict(X_train)
    print("Predicting on test set...")
    y_test_pred = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nBest Degree: {best_degree}")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Testing Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    q2_main()
