import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import log2


# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, names=names)

# Map diagnosis to binary values
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Split into features and labels
X = data.iloc[:, 2:]
y = data['Diagnosis']

# Split the data: 70% training, 10% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp)


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(counts[i] / np.sum(counts)) * log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

def info_gain(data, split_attribute_name, target_name="Diagnosis"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    # Calculate the values and the corresponding counts for the split attribute 
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    # Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def id3(data, originaldata, features, target_attribute_name="Diagnosis", parent_node_class = None):
    # Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    
    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    
    # If the feature space is empty, return the mode target feature value of the direct parent node
    elif len(features) == 0:
        return parent_node_class
    
    # If none of the above conditions holds true, grow the tree!
    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Select the feature which best splits the dataset
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features] # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information gain
        tree = {best_feature:{}}
        
        # Remove the feature with the best info gain from the feature space
        features = [i for i in features if i != best_feature]
        
        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # Call the ID3 algorithm for each of those sub_datasets with the new parameters
            subtree = id3(sub_data, data, features, target_attribute_name, parent_node_class)
            
            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
        
        return tree

def predict(query, tree, default = 1):
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]
            #3.
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result

def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i], tree, 1.0) 
    accuracy = np.sum(predicted["predicted"] == data["Diagnosis"]) / len(data)
    return accuracy

# Train the ID3 tree
tree = id3(X_train.join(y_train), X_train.join(y_train), X_train.columns)

# Test the tree
train_acc = test(X_train.join(y_train), tree)
val_acc = test(X_val.join(y_val), tree)
test_acc = test(X_test.join(y_test), tree)

print(f"Train Accuracy: {train_acc}")
print(f"Validation Accuracy: {val_acc}")
print(f"Test Accuracy: {test_acc}")

