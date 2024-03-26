"""Models for the datasets."""
from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class NN(nn.Module):
    """NN for Adult dataset."""

    def __init__(self:Self, in_shape:int, num_classes:int=10) -> None:
        """Initialize the model.

        Args:
        ----
            in_shape (int): The input shape.
            num_classes (int, optional): The number of classes. Defaults to 10.

        """
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self:Self, inputs:torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        outputs = F.relu(self.fc1(inputs))
        outputs = F.relu(self.fc2(outputs))
        return F.relu(self.fc3(outputs))

class ConvNet(nn.Module):
    """Convolutional Neural Network model."""

    def __init__(self:Self) -> None:
        """Initialize the ConvNet model."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self:Self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)





