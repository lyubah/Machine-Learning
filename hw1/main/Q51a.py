# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from classifiers import * 
from main import * 


def run_experiments():
    train_loader, valid_loader = load_data('binary')

    classifiers = {
        'Perceptron': PerceptronClassifier(784),
        'Passive Aggressive': PassiveAggressiveClassifier(784),
    }

    for label, classifier in classifiers.items():
        print(f"Training {label}...")
        mistakes = train_classifier(classifier, train_loader, 50)  # Training for 50 full passes
        plot_learning_curve(mistakes, label)

def plot_learning_curve(mistakes, label):
    plt.figure()
    plt.plot(range(1, 51), mistakes, marker='o')
    plt.title(f'Online Learning Curve for {label}')
    plt.xlabel('Training Iterations')
    plt.ylabel('Number of Mistakes')
    plt.legend()
    plt.savefig(f'{label}_online_learning_curve.png')
    plt.close()

# Call to run experiments
run_experiments()
