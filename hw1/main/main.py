import os
import matplotlib.pyplot as plt
from bin_classifiers import *
from multi_class import * 

# Graphing function
def graph_results(x, y, title, xlabel, ylabel, color='blue', linestyle='-', marker='o', save_path=None):
   
    plt.figure(figsize=(8, 6))  # Set figure size for better readability
    plt.plot(x, y, color=color, linestyle=linestyle, marker=marker, label=ylabel)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)  # Add gridlines
    plt.legend()  # Add legend for the y-axis label
    plt.tight_layout()  # Adjust layout so that everything fits nicely
    
    # Save the graph to the current directory if save_path is provided
    if save_path:
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.getcwd(), save_path)  # Save to current working directory
        plt.savefig(save_path)
    
    plt.show()  # Display the graph


def run_and_plot(algorithm_name: str, update_function: callable, num_iterations: int, use_avg: bool, file_prefix: str) -> None:
    
    results = run_algorithm(iteration=num_iterations, update_fn=update_function, dataset_type='train', use_averaging=use_avg)
    
    # Extract accuracies
    training_accuracy = results[2]
    testing_accuracy = results[3]
    
    # Plot training accuracy
    graph_results(
        x=range(1, num_iterations + 1),
        y=training_accuracy,
        title=f"{algorithm_name} Training Accuracy",
        xlabel="Iteration",
        ylabel="Accuracy",
        save_path=f"{file_prefix}_training_accuracy.png"
    )
    
    # Plot testing accuracy
    graph_results(
        x=range(1, num_iterations + 1),
        y=testing_accuracy,
        title=f"{algorithm_name} Test Accuracy",
        xlabel="Iteration",
        ylabel="Accuracy",
        save_path=f"{file_prefix}_testing_accuracy.png"
    )


def plot_algorithm_mistakes() -> None:
   

    # Run standard Perceptron
    perc = run_algorithm(iteration=50, update_fn=perceptron_update, dataset_type='train', use_averaging=False)
    mistakes = perc[1]
    graph_results(
        x=range(1, 51),
        y=mistakes,
        title="Binary Perceptron Mistakes",
        xlabel="Iteration",
        ylabel="Mistakes",
        save_path="perceptron_mistakes.png"
    )

    # Run Passive-Aggressive Algorithm
    PA = run_algorithm(iteration=50, update_fn=passive_aggressive_update, dataset_type='train', use_averaging=False)
    mistakes = PA[1]
    graph_results(
        x=range(1, 51),
        y=mistakes,
        title="Binary PA Mistakes",
        xlabel="Iteration",
        ylabel="Mistakes",
        save_path="passive_aggressive_mistakes.png"
    )

    # Run Averaged Perceptron Algorithm
    AP = run_algorithm(iteration=50, update_fn=averaged_perceptron_update, dataset_type='train', use_averaging=True)
    mistakes = AP[1]
    graph_results(
        x=range(1, 51),
        y=mistakes,
        title="Binary Averaged Perceptron Mistakes",
        xlabel="Iteration",
        ylabel="Mistakes",
        save_path="averaged_perceptron_mistakes.png"
    )


def plot_algorithm_accuracies() -> None:
    

    run_and_plot(
        algorithm_name="Binary Perceptron",
        update_function=perceptron_update,
        num_iterations=20,
        use_avg=False,
        file_prefix="perceptron"
    )

    run_and_plot(
        algorithm_name="Binary PA",
        update_function=passive_aggressive_update,
        num_iterations=20,
        use_avg=False,
        file_prefix="pa"
    )

    run_and_plot(
        algorithm_name="Averaged Perceptron",
        update_function=averaged_perceptron_update,
        num_iterations=20,
        use_avg=True,
        file_prefix="averaged_perceptron"
    )


def plot_learning_curve(algorithm_name: str, increment_function: callable, num_iterations: int, size: int, file_prefix: str) -> None:
    
    results = increment_function(iteration=num_iterations, size=size)
    
    # Extract data sizes and test accuracies
    data_size = results[1]
    test_accuracy = results[2]
    
    # Plot the learning curve
    graph_results(
        x=data_size,
        y=test_accuracy,
        title=f"{algorithm_name} Learning Curve",
        xlabel="Number of Training Examples",
        ylabel="Test Accuracy",
        save_path=f"{file_prefix}_learning_curve.png"
    )


def plot_all_learning_curves() -> None:
    
    
    # Plot Perceptron learning curve
    plot_learning_curve(
        algorithm_name="Binary Perceptron",
        increment_function=perceptron_increment_run,
        num_iterations=20,
        size=100,
        file_prefix="perceptron"
    )

    # Plot Passive-Aggressive learning curve
    plot_learning_curve(
        algorithm_name="Binary PA",
        increment_function=passive_aggressive_increment_run,
        num_iterations=20,
        size=100,
        file_prefix="pa"
    )
    
    plot_learning_curve(
        algorithm_name="Averaged Perceptron",
        increment_function=averaged_perceptron_update,
        num_iterations=20,
        size =100,
        use_avg=True,
        file_prefix="averaged_perceptron"
    )
    
    
    


# # Graphing function
# def graph_results(x, y, title, xlabel, ylabel, color='blue', linestyle='-', marker='o', save_path=None):
    
#     plt.figure(figsize=(8, 6))  # Set figure size for better readability
#     plt.plot(x, y, color=color, linestyle=linestyle, marker=marker, label=ylabel)
#     plt.title(title, fontsize=14)
#     plt.xlabel(xlabel, fontsize=12)
#     plt.ylabel(ylabel, fontsize=12)
#     plt.grid(True)  # Add gridlines
#     plt.legend()  # Add legend for the y-axis label
#     plt.tight_layout()  # Adjust layout so that everything fits nicely
    
#     # Save the graph to the current directory if save_path is provided
#     if save_path:
#         if not os.path.isabs(save_path):
#             save_path = os.path.join(os.getcwd(), save_path)  # Save to current working directory
#         plt.savefig(save_path)
    
#     plt.show()  # Display the graph


def run_and_plot_multi(algorithm_name: str, update_function: callable, num_iterations: int, file_prefix: str) -> None:
    """
    General function to run and plot results for multi-class algorithms (Perceptron, PA).
    
    :param algorithm_name: Name of the algorithm (e.g., "Perceptron", "PA").
    :param update_function: Update function (e.g., perceptron or PA update).
    :param num_iterations: Number of iterations to run.
    :param file_prefix: Prefix for saving plots.
    """
    results = run_multi_algorithm(iteration=num_iterations, update_fn=update_function, dataset_type='train', num_classes=10)
    
    # Extract accuracies
    training_accuracy = results[2]
    testing_accuracy = results[3]
    
    # Plot training accuracy
    graph_results(
        x=range(1, num_iterations + 1),
        y=training_accuracy,
        title=f"{algorithm_name} Training Accuracy",
        xlabel="Iteration",
        ylabel="Accuracy",
        save_path=f"{file_prefix}_training_accuracy.png"
    )
    
    # Plot testing accuracy
    graph_results(
        x=range(1, num_iterations + 1),
        y=testing_accuracy,
        title=f"{algorithm_name} Test Accuracy",
        xlabel="Iteration",
        ylabel="Accuracy",
        save_path=f"{file_prefix}_testing_accuracy.png"
    )



def plot_multi_algorithm_mistakes() -> None:
    """
    Plot mistakes for multi-class Perceptron and Passive-Aggressive algorithms.
    """

    # Run multi-class Perceptron
    perc = run_perceptron(iteration=50)
    mistakes = perc[1]
    graph_results(
        x=range(1, 51),
        y=mistakes,
        title="Multi-class Perceptron Mistakes",
        xlabel="Iteration",
        ylabel="Mistakes",
        save_path="multi_perceptron_mistakes.png"
    )

    # Run multi-class Passive-Aggressive Algorithm
    PA = run_passive_aggressive(iteration=50)
    mistakes = PA[1]
    graph_results(
        x=range(1, 51),
        y=mistakes,
        title="Multi-class PA Mistakes",
        xlabel="Iteration",
        ylabel="Mistakes",
        save_path="multi_pa_mistakes.png"
    )


def plot_multi_algorithm_accuracies() -> None:
    """
    Compute and plot the accuracy of multi-class Perceptron and Passive-Aggressive algorithms
    on training and testing datasets.
    """

    run_and_plot_multi(
        algorithm_name="Multi-class Perceptron",
        update_function=update_multi_weights_perceptron,  # Updated function name
        num_iterations=20,
        file_prefix="multi_perceptron"
    )

    run_and_plot_multi(
        algorithm_name="Multi-class PA",
        update_function=update_multi_weights_pa,  # Updated function name
        num_iterations=20,
        file_prefix="multi_pa"
    )

def plot_multi_learning_curve(algorithm_name: str, increment_function: callable, num_iterations: int, size: int, file_prefix: str) -> None:
    
    results = increment_function(iteration=num_iterations, size=size)
    
    # Extract data sizes and test accuracies
    data_size = results[1]
    test_accuracy = results[2]
    
    # Plot the learning curve
    graph_results(
        x=data_size,
        y=test_accuracy,
        title=f"{algorithm_name} Learning Curve",
        xlabel="Number of Training Examples",
        ylabel="Test Accuracy",
        save_path=f"{file_prefix}_learning_curve.png"
    )


def plot_all_multi_learning_curves() -> None:
    
    # Plot Perceptron learning curve
    plot_multi_learning_curve(
        algorithm_name="Multi-class Perceptron",
        increment_function=increment_run,
        num_iterations=20,
        size=100,
        file_prefix="multi_perceptron"
    )

    # Plot Passive-Aggressive learning curve
    plot_multi_learning_curve(
        algorithm_name="Multi-class PA",
        increment_function=increment_run,
        num_iterations=20,
        size=100,
        file_prefix="multi_pa"
    )


def main():
    
    # # Run the learning curves for all binary classifiers
    # print("Running all binary learning curves...")
    # plot_all_learning_curves()
    
    
    # Run the multi-class algorithm accuracy plots
    print("Running multi-class algorithm accuracy plots...")
    plot_multi_algorithm_accuracies()

    # Run the multi-class algorithm mistakes plots
    print("Running multi-class algorithm mistakes plots...")
    plot_multi_algorithm_mistakes()

    # Run the learning curves for all multi-class classifiers
    print("Running all multi-class learning curves...")
    plot_all_multi_learning_curves()

   

if __name__ == "__main__":
    main()
