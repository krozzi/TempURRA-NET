import numpy as np


def calculate_loss(predicted_coordinates, ground_truth_labels):
    # Aligning ground truth labels with predicted coordinates
    aligned_labels = align_ground_truth(predicted_coordinates, ground_truth_labels)

    # Calculating Mean Squared Error
    loss = np.mean((predicted_coordinates - aligned_labels) ** 2)

    return loss


def align_ground_truth(predicted_coordinates, ground_truth_labels):
    # Decide on a strategy to align ground truth labels with predicted coordinates
    # For example, you can interpolate or resample ground truth labels

    # Example: Resampling ground truth labels to match the length of predicted coordinates
    aligned_labels = resample_ground_truth(predicted_coordinates, ground_truth_labels)

    return aligned_labels


def resample_ground_truth(predicted_coordinates, ground_truth_labels):
    # Resample ground truth labels to match the length of predicted coordinates
    # Example: Linear interpolation
    aligned_labels = np.interp(np.linspace(0, len(ground_truth_labels) - 1, len(predicted_coordinates)),
                               np.arange(len(ground_truth_labels)),
                               ground_truth_labels)

    return aligned_labels


# Example usage
predicted_coordinates = np.random.rand(179)  # Example predicted coordinates
ground_truth_labels = predicted_coordinates[:151]  # Example ground truth labels (variable length)
print(predicted_coordinates)
print(ground_truth_labels)
loss = calculate_loss(predicted_coordinates, ground_truth_labels)
print("Loss:", loss)