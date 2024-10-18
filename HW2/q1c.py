import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import fashion_mnist
import warnings
import os
from sklearn.preprocessing import MinMaxScaler


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_and_split_data(test_size=0.2, validation_size=0.2, random_state=42):
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    
    # First split: Train + Validation vs Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_train_full, y_train_full, test_size=test_size, random_state=random_state, stratify=y_train_full
    )
    
    # Second split: Train vs Validation
    val_relative_size = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(X_train, X_val, X_test):
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test


def check_overlaps(X_train, X_val, X_test):
    # Convert to sets of tuples for comparison
    train_set = set(map(tuple, X_train))
    val_set = set(map(tuple, X_val))
    test_set = set(map(tuple, X_test))

    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    print(f"Overlap between Train and Validation: {len(overlap_train_val)} samples")
    print(f"Overlap between Train and Test: {len(overlap_train_test)} samples")
    print(f"Overlap between Validation and Test: {len(overlap_val_test)} samples")


# def preprocess_data(X_train, X_val, X_test):
#     # Normalize data to [0, 1] range
#     scaler = MinMaxScaler()
#     X_train = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1)))
#     X_val = scaler.transform(X_val.reshape((X_val.shape[0], -1)))
#     X_test = scaler.transform(X_test.reshape((X_test.shape[0], -1)))
#     return X_train, X_val, X_test



def train_svm_with_poly_kernel(X_train, y_train, C, X_val, y_val, X_test, y_test, degree):
    print(f"Training SVM with polynomial kernel of degree={degree}, C={C}...")
    if degree == 1:
        kernel_type = 'linear'  # Treat polynomial of degree 1 as linear kernel
    else:
        kernel_type = 'poly'

    clf = svm.SVC(kernel=kernel_type, C=C, degree=degree, random_state=12, max_iter=10000, tol=1e-2, verbose=True)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    num_support_vectors = len(clf.support_)

    return train_acc, val_acc, test_acc, num_support_vectors


# def train_svm_with_poly_kernel(X_train, y_train, C, X_val, y_val, X_test, y_test, degree):
#     """
#     Trains an SVM with a polynomial kernel of the specified degree and returns accuracies and support vectors.
#     """
#     print(f"Training SVM with polynomial kernel of degree={degree}, C={C}...")
#     # clf = svm.SVC(kernel='poly', C=C, degree=degree, random_state=12, max_iter=10000)
#     # cache_size=2000,
#     clf = svm.SVC(kernel='poly', C=C, degree=degree, random_state=12, max_iter=1000, tol=1e-2, verbose=True)

#     clf.fit(X_train, y_train)
    
#     y_train_pred = clf.predict(X_train)
#     y_val_pred = clf.predict(X_val)
#     y_test_pred = clf.predict(X_test)

#     train_acc = accuracy_score(y_train, y_train_pred)
#     val_acc = accuracy_score(y_val, y_val_pred)
#     test_acc = accuracy_score(y_test, y_test_pred)
#     num_support_vectors = clf.n_support_.sum()

#     return train_acc, val_acc, test_acc, num_support_vectors


# def plot_poly_kernel_comparison(degrees, accuracies, support_vectors):
#     """
#     Plots accuracies and support vectors for different polynomial kernels.
#     """
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     sns.set(style="whitegrid")

#     ax1.set_xlabel('Polynomial Degree')
#     ax1.set_ylabel('Accuracy', color='tab:blue')
#     ax1.plot(degrees, [acc[0] for acc in accuracies], 'o-', label='Training Accuracy', color='tab:blue')
#     ax1.plot(degrees, [acc[1] for acc in accuracies], 's-', label='Validation Accuracy', color='tab:green')
#     ax1.plot(degrees, [acc[2] for acc in accuracies], '^-', label='Testing Accuracy', color='tab:red')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')
#     ax1.legend(loc='upper left')

#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Number of Support Vectors', color='tab:purple')
#     ax2.plot(degrees, support_vectors, 'D-', color='tab:purple', label='Support Vectors')
#     ax2.tick_params(axis='y', labelcolor='tab:purple')
#     ax2.legend(loc='upper right')

#     fig.tight_layout()
#     plt.title('Comparison of Polynomial Kernels (Degree 2, 3, 4)')
#     plt.show()

def plot_support_vectors_poly_kernel_comparison(degrees, support_vectors):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")

    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Number of Support Vectors', color='tab:purple')
    ax.plot(degrees, support_vectors, 'D-', color='tab:purple', label='Support Vectors')
    ax.tick_params(axis='y', labelcolor='tab:purple')
    ax.legend(loc='upper right')

    plt.title('Support Vectors Across Polynomial Degrees')
    fig.savefig(os.path.join(os.getcwd(), 'SVM_Support_Vectors_Polynomial_Degrees.png'))
    plt.show()

def plot_accuracies_poly_kernel_comparison(degrees, accuracies):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")

    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Accuracy', color='tab:blue')
    ax.plot(degrees, [acc[0] for acc in accuracies], 'o-', label='Training Accuracy', color='tab:blue')
    ax.plot(degrees, [acc[1] for acc in accuracies], 's-', label='Validation Accuracy', color='tab:green')
    ax.plot(degrees, [acc[2] for acc in accuracies], '^-', label='Testing Accuracy', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.legend(loc='upper left')

    plt.title('Accuracy Comparison Across Polynomial Degrees')
    fig.savefig(os.path.join(os.getcwd(), 'SVM_Accuracy_Polynomial_Degrees.png'))
    plt.show()


def q1c_main():
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)

    # Confirm data splits
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Check for overlaps
    check_overlaps(X_train, X_val, X_test)

    # Use the best C from part (a)
    C_best = 0.01
    degrees = [1, 2, 3, 4]  # Include linear (degree=1) for comparison

    accuracies = []
    support_vectors = []

    for degree in degrees:
        train_acc, val_acc, test_acc, num_sv = train_svm_with_poly_kernel(
            X_train, y_train, C_best, X_val, y_val, X_test, y_test, degree
        )
        print(f"Degree {degree}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}, Support Vectors={num_sv}")
        accuracies.append((train_acc, val_acc, test_acc))
        support_vectors.append(num_sv)

    # Plot accuracies and support vectors
    plot_accuracies_poly_kernel_comparison(degrees, accuracies)
    plot_support_vectors_poly_kernel_comparison(degrees, support_vectors)



if __name__ == "__main__":
    q1c_main()
