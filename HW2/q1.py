# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.datasets import fashion_mnist
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix



# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_split_data(test_size=0.2, validation_size=0.2, random_state=42):
    """
    Loads the Fashion MNIST dataset and splits it into training, validation, and testing sets.
    """
    print("Loading Fashion MNIST data...")
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
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_val, X_test):
    """
    Flattens and scales the data.
    """
    print("Preprocessing data...")
    # Flatten the images
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit on training data and transform
    X_train = scaler.fit_transform(X_train)
    
    # Transform validation and test data
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test


def train_svm_linear(X_train, y_train, C_values, X_val, y_val, X_test, y_test):
    """
    Trains linear SVMs with different C values and records accuracies and support vectors.
    """
    training_accuracies = []
    validation_accuracies = []
    testing_accuracies = []
    num_support_vectors = []
    
    for C in C_values:
        print(f"Training SVM with C={C}...")
        clf = svm.SVC(kernel='linear', C=C, random_state=12, max_iter=10000, tol=1e-4, verbose=True) # I added tol because it was not converging
        
        clf.fit(X_train, y_train)
        
        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)
        
        # Compute accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        training_accuracies.append(train_acc)
        validation_accuracies.append(val_acc)
        testing_accuracies.append(test_acc)
        
        # Number of support vectors
        total_sv = clf.n_support_.sum()
        num_support_vectors.append(total_sv)
        
        print(f"C={C}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}, Support Vectors={total_sv}")
    
    return training_accuracies, validation_accuracies, testing_accuracies, num_support_vectors



def plot_accuracies(C_values, training_accuracies, validation_accuracies, testing_accuracies):
    """
    Plots training, validation, and testing accuracies as a function of C.
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    plt.plot(C_values, training_accuracies, marker='o', label='training Accuracy')
    plt.plot(C_values, validation_accuracies, marker='s', label='validation Accuracy')
    plt.plot(C_values, testing_accuracies, marker='^', label='testing Accuracy')
    
    plt.xscale('log')
    plt.xlabel('C Value (log scale)')
    plt.ylabel('Accuracy')
    plt.title('SVM Accuracy vs C Value (Linear Kernel)')
    plt.legend()

    plt.savefig(os.path.join(os.getcwd(), 'SVM_Accuracy_vs_C_Value_1.png'))
    plt.show()

def plot_support_vectors(C_values, num_support_vectors):
    """
    Plots the number of support vectors as a function of C.
    """
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    plt.plot(C_values, num_support_vectors, marker='D', color='purple')
    plt.xscale('log')
    plt.xlabel('C Value (log scale)')
    plt.ylabel('Number of Support Vectors')
    plt.title('Number of Support Vectors vs C Value (Linear Kernel)')
    
    plt.savefig(os.path.join(os.getcwd(), 'Support_Vectors_vs_C_Value_1.png'))
    plt.show()
    
    
def plot_confusion_matrix(conf_matrix, labels, filename='confusion_matrix.pdf'):
    """
    Plots and saves the confusion matrix as a PDF file.
    """
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Save the plot as a PDF file
    plt.savefig(os.path.join(os.getcwd(), filename), format='pdf')
    plt.show()


# Ensure main function and others are unchanged
def one_b(X_train, X_val, y_train, y_val, X_test, y_test, C_best):
    # Combine the training and validation sets
    X_train_combined = np.vstack((X_train, X_val))
    y_train_combined = np.hstack((y_train, y_val))

    # Train the SVM with the best C on the combined data
    svm_best = svm.SVC(kernel='linear', C=C_best)
    svm_best.fit(X_train_combined, y_train_combined)

    # Evaluate on the test set
    y_test_pred = svm_best.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    print(f"Testing Accuracy: {accuracy_test}")
    print("Confusion Matrix (10x10):\n", conf_matrix)

    return accuracy_test, conf_matrix





def q1_main():
    
    C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    
    training_accuracies, validation_accuracies, testing_accuracies, num_support_vectors = train_svm_linear(
        X_train, y_train, C_values, X_val, y_val, X_test, y_test
    )
    
    C_best_index = validation_accuracies.index(max(validation_accuracies))  
    C_best = C_values[C_best_index]  
    best_n_support_linear = num_support_vectors[C_best_index]   
    
    # Print the results
    print("Best validation C:", C_best)
    print("Number of support vectors for best C:", best_n_support_linear)
    
    ############################################
    """
    The results:
    Best validation C: 0.01
    Number of support vectors for best C: 12250

    """
    ################################
    
    plot_accuracies(C_values, training_accuracies, validation_accuracies, testing_accuracies)
    plot_support_vectors(C_values, num_support_vectors)
    
    accuracy_test, conf_matrix = one_b(X_train, X_val, y_train, y_val, X_test, y_test, C_best)
    # Define the labels (if it's Fashion MNIST, for example, you can use class names)
    labels = [str(i) for i in range(10)]  # Labels for Fashion MNIST (0-9)
    
    # Plot and save the confusion matrix as PDF
    plot_confusion_matrix(conf_matrix, labels, filename='confusion_matrix.pdf')

    

if __name__ == "__main__":
    q1_main()
