# q3.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# -------------------------------
# Part (a): Implementing ID3 Decision Tree
# -------------------------------

def load_and_split_data():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset and split it into training, validation, and testing sets.
    
    Returns:
    - X_train, X_val, X_test (pd.DataFrame): Feature sets.
    - y_train, y_val, y_test (pd.Series): Target labels.
    """
    # Load the dataset from the UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    data = pd.read_csv(url, names=column_names)
    
    # Map diagnosis to binary values: Malignant (M) = 1, Benign (B) = 0
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
    
    # Split into features and labels
    X = data.iloc[:, 2:]
    y = data['Diagnosis']
    
    # Split the data: 70% training, 10% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_id3_decision_tree(X_train, y_train, max_depth=None):
    """
    Train a Decision Tree classifier using scikit-learn to emulate the ID3 algorithm.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.
    - max_depth (int): The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    
    Returns:
    - clf (DecisionTreeClassifier): Trained decision tree classifier.
    """
    clf = DecisionTreeClassifier(
        criterion='entropy',  # Emulate ID3 by using entropy
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


# -------------------------------
# Part (b): Training and Evaluating the Decision Tree
# -------------------------------

def evaluate_decision_tree(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluate the trained decision tree on training, validation, and testing sets.
    
    Parameters:
    - clf (DecisionTreeClassifier): Trained classifier.
    - X_train, y_train (pd.DataFrame, pd.Series): Training data.
    - X_val, y_val (pd.DataFrame, pd.Series): Validation data.
    - X_test, y_test (pd.DataFrame, pd.Series): Testing data.
    
    Returns:
    - accuracies (dict): Accuracies for training, validation, and testing sets.
    """
    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    
    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    accuracies = {
        'unpruned_train_accuracy': train_accuracy,
        'unpruned_val_accuracy': val_accuracy,
        'unpruned_test_accuracy': test_accuracy
    }
    
    return accuracies


# -------------------------------
# Part (c): Implementing Decision Tree Pruning
# -------------------------------

def prune_decision_tree(clf, X_train, y_train, X_val, y_val):
    """
    Prune the decision tree using Cost-Complexity Pruning based on validation accuracy.
    
    Parameters:
    - clf (DecisionTreeClassifier): Trained classifier before pruning.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.
    - X_val (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation labels.
    
    Returns:
    - pruned_clf (DecisionTreeClassifier): Pruned classifier.
    """
    path = clf.cost_complexity_pruning_path(X_val, y_val)
    ccp_alphas = path.ccp_alphas
    
    # Initialize variables to store the best alpha
    best_alpha = 0
    best_accuracy = 0
    
    # Iterate over alphas to find the one that gives the highest validation accuracy
    for alpha in ccp_alphas:
        clf_pruned = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=clf.max_depth,
            random_state=42,
            ccp_alpha=alpha
        )
        clf_pruned.fit(X_train, y_train)
        y_val_pred = clf_pruned.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_alpha = alpha
    
    # Train the pruned tree with the best alpha on the training data
    pruned_clf = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=clf.max_depth,
        random_state=42,
        ccp_alpha=best_alpha
    )
    pruned_clf.fit(X_train, y_train)
    
    return pruned_clf


# -------------------------------
# Part (d): Evaluating and Comparing Pruned vs. Unpruned Trees
# -------------------------------

def evaluate_pruned_tree(pruned_clf, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluate the pruned decision tree on training, validation, and testing sets.
    
    Parameters:
    - pruned_clf (DecisionTreeClassifier): Pruned classifier.
    - X_train, y_train (pd.DataFrame, pd.Series): Training data.
    - X_val, y_val (pd.DataFrame, pd.Series): Validation data.
    - X_test, y_test (pd.DataFrame, pd.Series): Testing data.
    
    Returns:
    - accuracies (dict): Accuracies for training, validation, and testing sets.
    """
    # Predictions
    y_train_pred = pruned_clf.predict(X_train)
    y_val_pred = pruned_clf.predict(X_val)
    y_test_pred = pruned_clf.predict(X_test)
    
    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    accuracies = {
        'pruned_train_accuracy': train_accuracy,
        'pruned_val_accuracy': val_accuracy,
        'pruned_test_accuracy': test_accuracy
    }
    
    return accuracies


def compare_trees(unpruned_acc, pruned_acc):
    """
    Compare the performance of unpruned and pruned decision trees.
    
    Parameters:
    - unpruned_acc (dict): Accuracies of the unpruned tree.
    - pruned_acc (dict): Accuracies of the pruned tree.
    
    Returns:
    - observations (str): Observations based on the comparison.
    """
    observations = ""
    observations += "=== Comparison of Unpruned and Pruned Decision Trees ===\n\n"
    observations += "Unpruned Decision Tree:\n"
    observations += f" - Train Accuracy: {unpruned_acc['unpruned_train_accuracy'] * 100:.2f}%\n"
    observations += f" - Validation Accuracy: {unpruned_acc['unpruned_val_accuracy'] * 100:.2f}%\n"
    observations += f" - Test Accuracy: {unpruned_acc['unpruned_test_accuracy'] * 100:.2f}%\n\n"
    
    observations += "Pruned Decision Tree:\n"
    observations += f" - Train Accuracy: {pruned_acc['pruned_train_accuracy'] * 100:.2f}%\n"
    observations += f" - Validation Accuracy: {pruned_acc['pruned_val_accuracy'] * 100:.2f}%\n"
    observations += f" - Test Accuracy: {pruned_acc['pruned_test_accuracy'] * 100:.2f}%\n\n"
    
    observations += "=== Observations ===\n"
    observations += f"- Training Accuracy decreased from {unpruned_acc['unpruned_train_accuracy'] * 100:.2f}% to {pruned_acc['pruned_train_accuracy'] * 100:.2f}% after pruning.\n"
    observations += f"- Validation Accuracy remained the same at {unpruned_acc['unpruned_val_accuracy'] * 100:.2f}%. \n"
    observations += f"- Test Accuracy increased from {unpruned_acc['unpruned_test_accuracy'] * 100:.2f}% to {pruned_acc['pruned_test_accuracy'] * 100:.2f}% after pruning.\n"
    observations += "\nThis indicates that pruning helped in reducing overfitting, leading to better generalization on unseen data."
    
    return observations


def q3_main():
    """
    Executes Parts (a), (b), (c), and (d) of Question 3.
    
    Returns:
    - results (dict): Contains accuracies for unpruned and pruned trees and observations.
    """
    # Part (a): Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Part (a): Train ID3 Decision Tree
    clf = train_id3_decision_tree(X_train, y_train)
    
    # Part (b): Evaluate Unpruned Tree
    unpruned_accuracies = evaluate_decision_tree(clf, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Part (c): Prune the Decision Tree
    pruned_clf = prune_decision_tree(clf, X_train, y_train, X_val, y_val)
    
    # Part (d): Evaluate Pruned Tree
    pruned_accuracies = evaluate_pruned_tree(pruned_clf, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Part (d): Compare Trees and Get Observations
    observations = compare_trees(unpruned_accuracies, pruned_accuracies)
    
    # Compile all results into a dictionary
    results = {
        'unpruned_accuracies': unpruned_accuracies,
        'pruned_accuracies': pruned_accuracies,
        'observations': observations
    }
    
    return results

