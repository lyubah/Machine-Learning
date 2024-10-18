# main.py
# -*- coding: utf-8 -*-

import q1
import q1c
import q2
import q3  # Import the q3 module

def main():
    """
    Main script to execute Parts 1(a), 1(b), 1(c), Part 2, and Question 3 sequentially.
    """
    print("=== Executing Part 1(a) and Part 1(b) ===")
    # Execute Part 1(a) and Part 1(b)
    C_best, accuracy_test, conf_matrix, X_train, X_val, X_test, y_train, y_val, y_test = q1.q1_main()
    
    print("\n=== Executing Part 1(c) ===")
    # Execute Part 1(c) using the best C and existing data splits
    best_degree = q1c.q1c_main(C_best, X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n=== Executing Part 2 ===")
    # Execute Part 2 using the best_degree from Part 1(c)
    train_acc, val_acc, test_acc, mistakes_per_iteration = q2.q2_main(best_degree, X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\n=== Executing Question 3 ===")
    # Execute Question 3 (Breast Cancer Classifier using Decision Trees)
    q3.q3_main()
    
    print("\n=== Summary of Results ===")
    print(f"Best C from Part 1(a): {C_best}")
    print(f"Test Accuracy from Part 1(b): {accuracy_test * 100:.2f}%")
    print(f"Best Polynomial Degree from Part 1(c): {best_degree}")
    print(f"Training Accuracy from Part 2: {train_acc * 100:.2f}%")
    print(f"Validation Accuracy from Part 2: {val_acc * 100:.2f}%")
    print(f"Testing Accuracy from Part 2: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
