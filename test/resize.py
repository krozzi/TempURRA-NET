import torch


def reshape_tensor(input_tensor, target_shape):
    """
    Reshapes the input tensor to the target shape without adding junk data.

    Parameters:
        input_tensor (torch.Tensor): Input tensor to be reshaped.
        target_shape (tuple): Target shape for the tensor.

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    # Calculate the number of times to repeat the tensor completely
    num_repeats = target_shape[0] // input_tensor.shape[0]

    # Calculate the remaining portion
    remainder = target_shape[0] % input_tensor.shape[0]

    # Repeat the tensor completely
    new_tensor = torch.cat([input_tensor] * num_repeats, dim=0)

    # Append the remaining portion
    if remainder > 0:
        new_tensor = torch.cat([new_tensor, input_tensor[:remainder]], dim=0)

    return new_tensor


# Example usage:
input_tensor = torch.randn(6, 512, 4, 8)
target_shape = (32, )
new_tensor = reshape_tensor(input_tensor, target_shape)
print(new_tensor.shape)
