# q3.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import log2
import warnings
warnings.filterwarnings('ignore')


# -------------------------------
# Part (a): Implementing ID3 Decision Tree
# -------------------------------

def entropy(target_col):
    """
    Calculate the entropy of a dataset.

    Parameters:
    - target_col (pd.Series): The target labels.

    Returns:
    - float: Entropy value.
    """
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_val = -np.sum([
        (counts[i] / np.sum(counts)) * log2(counts[i] / np.sum(counts))
        for i in range(len(elements))
    ])
    return entropy_val


def info_gain_continuous(data, split_attribute_name, target_name="Diagnosis"):
    """
    Calculate the information gain for a continuous feature by finding the best threshold.

    Parameters:
    - data (pd.DataFrame): The dataset.
    - split_attribute_name (str): The feature to split on.
    - target_name (str): The target attribute name.

    Returns:
    - tuple: (best_information_gain, best_threshold)
    """
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    # Sort the data based on the split attribute
    data_sorted = data.sort_values(split_attribute_name)
    values = data_sorted[split_attribute_name].unique()

    # If there's only one unique value, no split is possible
    if len(values) == 1:
        return 0, None

    # Compute candidate thresholds
    thresholds = (values[:-1] + values[1:]) / 2

    # Initialize variables to track the best threshold
    best_threshold = None
    best_info_gain = -1

    for threshold in thresholds:
        # Split the data
        left_split = data_sorted[data_sorted[split_attribute_name] <= threshold]
        right_split = data_sorted[data_sorted[split_attribute_name] > threshold]

        # Calculate the weighted entropy
        left_entropy = entropy(left_split[target_name])
        right_entropy = entropy(right_split[target_name])
        weighted_entropy = (len(left_split) / len(data_sorted)) * left_entropy + \
                           (len(right_split) / len(data_sorted)) * right_entropy

        # Calculate information gain
        information_gain = total_entropy - weighted_entropy

        # Update the best information gain and threshold
        if information_gain > best_info_gain:
            best_info_gain = information_gain
            best_threshold = threshold

    return best_info_gain, best_threshold


def id3(data, originaldata, features, target_attribute_name="Diagnosis", parent_node_class=None):
    """
    Build the decision tree using the ID3 algorithm.

    Parameters:
    - data (pd.DataFrame): The current dataset.
    - originaldata (pd.DataFrame): The original dataset.
    - features (list): List of features to consider for splitting.
    - target_attribute_name (str): The target attribute name.
    - parent_node_class (int): The class of the parent node.

    Returns:
    - dict or int: The decision tree represented as nested dictionaries or a leaf node.
    """
    # Stopping criteria
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(
            np.unique(originaldata[target_attribute_name], return_counts=True)[1]
        )]

    elif len(features) == 0:
        return parent_node_class

    else:
        # Set the default value for this node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(
            np.unique(data[target_attribute_name], return_counts=True)[1]
        )]

        # Initialize variables to track the best feature and threshold
        best_info_gain = -1
        best_feature = None
        best_threshold = None

        # Iterate over all features to find the best split
        for feature in features:
            info_gain_val, threshold = info_gain_continuous(data, feature, target_attribute_name)
            if info_gain_val > best_info_gain:
                best_info_gain = info_gain_val
                best_feature = feature
                best_threshold = threshold

        # If no information gain is achieved, return the parent node's class
        if best_info_gain == 0 or best_feature is None:
            return parent_node_class

        # Create the tree structure with the best feature and threshold
        tree = {best_feature: {"threshold": best_threshold, "<=": {}, ">": {}}}

        # Split the dataset based on the best threshold
        left_split = data[data[best_feature] <= best_threshold]
        right_split = data[data[best_feature] > best_threshold]

        # Remove the best feature from the list for further splits
        features = [f for f in features if f != best_feature]

        # Recursively build the tree for left and right splits
        tree[best_feature]["<="] = id3(left_split, data, features, target_attribute_name, parent_node_class)
        tree[best_feature][">"] = id3(right_split, data, features, target_attribute_name, parent_node_class)

        return tree


# -------------------------------
# Part (b): Training and Evaluating the Decision Tree
# -------------------------------

def predict(query, tree, default=1):
    """
    Predict the class label for a single instance using the decision tree.

    Parameters:
    - query (dict): A single instance with feature values.
    - tree (dict or int): The decision tree.
    - default (int): Default class label if prediction fails.

    Returns:
    - int: Predicted class label.
    """
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                feature_value = query[key]
                subtree = tree[key]
                threshold = subtree["threshold"]
                if feature_value <= threshold:
                    result = subtree["<="]
                else:
                    result = subtree[">"]
            except:
                return default

            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result
    return default


def test(data, tree):
    """
    Test the decision tree on a dataset and calculate accuracy.

    Parameters:
    - data (pd.DataFrame): The dataset with features and target.
    - tree (dict or int): The decision tree.

    Returns:
    - float: Accuracy of the decision tree on the dataset.
    """
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1)

    accuracy = np.sum(predicted["predicted"] == data["Diagnosis"]) / len(data)
    return accuracy


# -------------------------------
# Part (c): Implementing Decision Tree Pruning
# -------------------------------

def get_majority_class(data):
    """
    Get the majority class label from the dataset.

    Parameters:
    - data (pd.DataFrame): The dataset.

    Returns:
    - int: Majority class label.
    """
    return data['Diagnosis'].mode()[0]


def prune(tree, X_val, y_val):
    """
    Prune the decision tree using the validation set.

    Parameters:
    - tree (dict or int): The decision tree.
    - X_val (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation labels.

    Returns:
    - dict or int: Pruned decision tree.
    """
    # Combine X_val and y_val for easy access
    data_val = X_val.copy()
    data_val['Diagnosis'] = y_val

    def prune_node(node, data):
        if not isinstance(node, dict):
            return node

        # Extract the feature, threshold, and subtrees
        feature = list(node.keys())[0]
        subtree = node[feature]
        threshold = subtree["threshold"]

        # Split the data based on the current node's feature and threshold
        left_split = data[data[feature] <= threshold]
        right_split = data[data[feature] > threshold]

        # Prune the left subtree
        subtree["<="] = prune_node(subtree["<="], left_split)

        # Prune the right subtree
        subtree[">"] = prune_node(subtree[">"], right_split)

        # After pruning subtrees, check if this node can be pruned
        # Predict using the current subtree
        current_predictions = data.apply(lambda row: predict(row, tree, get_majority_class(data)), axis=1)
        current_accuracy = np.sum(current_predictions == data['Diagnosis']) / len(data)

        # Predict using majority class
        majority_class = get_majority_class(data)
        majority_predictions = np.full(len(data), majority_class)
        majority_accuracy = np.sum(majority_predictions == data['Diagnosis']) / len(data)

        # If majority accuracy is better or equal, prune this node
        if majority_accuracy >= current_accuracy:
            return majority_class
        else:
            return subtree

    pruned_tree = prune_node(tree, data_val)
    return pruned_tree


# -------------------------------
# Part (d): Evaluating and Comparing Pruned vs. Unpruned Trees
# -------------------------------

def visualize_tree(tree, depth=0):
    """
    Visualize the structure of the decision tree in the console.

    Parameters:
    - tree (dict or int): The decision tree.
    - depth (int): Current depth for indentation.
    """
    indent = "  " * depth
    if not isinstance(tree, dict):
        print(f"{indent}=> Class {tree}")
        return
    feature = list(tree.keys())[0]
    threshold = tree[feature]["threshold"]
    print(f"{indent}{feature} <= {threshold}")
    visualize_tree(tree[feature]["<="], depth + 1)
    print(f"{indent}{feature} > {threshold}")
    visualize_tree(tree[feature][">"], depth + 1)


# -------------------------------
# Main Function to Execute All Parts
# -------------------------------

def q3_main():
    """
    Executes Parts (a), (b), (c), and (d) of Question 3.
    """
    print("=== Part (a): Implementing ID3 Decision Tree ===")
    # Train the ID3 tree
    tree = id3(X_train.join(y_train), X_train.join(y_train), X_train.columns.tolist())
    print("\nID3 Decision Tree constructed.")

    print("\n=== Part (b): Evaluating the Decision Tree ===")
    # Test the tree
    train_acc = test(X_train.join(y_train), tree)
    val_acc = test(X_val.join(y_val), tree)
    test_acc = test(X_test.join(y_test), tree)

    print(f"Unpruned Tree - Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Unpruned Tree - Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Unpruned Tree - Test Accuracy: {test_acc * 100:.2f}%")

    print("\n=== Part (c): Implementing Decision Tree Pruning ===")
    # Prune the tree using the validation set
    pruned_tree = prune(tree, X_val.join(y_val), y_val)
    print("Decision tree pruning completed.")

    print("\n=== Part (d): Evaluating Pruned vs. Unpruned Trees ===")
    # Test the pruned tree
    pruned_train_acc = test(X_train.join(y_train), pruned_tree)
    pruned_val_acc = test(X_val.join(y_val), pruned_tree)
    pruned_test_acc = test(X_test.join(y_test), pruned_tree)

    print(f"\nPruned Tree - Train Accuracy: {pruned_train_acc * 100:.2f}%")
    print(f"Pruned Tree - Validation Accuracy: {pruned_val_acc * 100:.2f}%")
    print(f"Pruned Tree - Test Accuracy: {pruned_test_acc * 100:.2f}%")

    # Summary of Results
    print("\n=== Summary of Results ===")
    print(f"Unpruned Tree - Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Unpruned Tree - Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Unpruned Tree - Test Accuracy: {test_acc * 100:.2f}%\n")

    print(f"Pruned Tree - Train Accuracy: {pruned_train_acc * 100:.2f}%")
    print(f"Pruned Tree - Validation Accuracy: {pruned_val_acc * 100:.2f}%")
    print(f"Pruned Tree - Test Accuracy: {pruned_test_acc * 100:.2f}%")

    # Visualization of the Pruned Tree
    print("\n=== Visualization of the Pruned Decision Tree ===")
    visualize_tree(pruned_tree)


# -------------------------------
# End of q3.py
# -------------------------------

