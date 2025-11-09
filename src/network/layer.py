import math
import random

class Layer:
    def __init__(
            self,
            number_of_inputs: int,
            number_of_outputs: int
    ) -> None:
        self.number_of_inputs: int = number_of_inputs
        self.number_of_outputs: int = number_of_outputs

        # Parameters.
        self.weights: list[list[float]]
        self.biases: list[float]
        # NOTE: Weights and its loss gradients are stored in the format of
        # [neuronIn][neuronOut] as in modern neural network libraries.

        # Loss gradients.
        self.loss_gradient_weights: list[list[float]]
        self.loss_gradient_biases: list[float]

        # Metadata stored for back propagation.
        self.inputs: list[float]
        self.weighted_inputs: list[float]
        self.activations: list[float]

        self.initialize_parameters()
    
    # Main methods.
    def feed_forward(self, inputs: list[float]) -> list[float]:
        # NOTE: We are assuming that the user knows what he's doing and will
        # be smart enough to understand about the correct types from the type
        # annotations. This class is also internal and not supposed to be used
        # outside.
        
        self.inputs = inputs.copy()
        self.activations = []

        for neuron_out in range(self.number_of_outputs):
            weighted_input: float = self.biases[neuron_out]
            for neuron_in in range(self.number_of_inputs):
                weighted_input += self.inputs[neuron_in] * self.weights[neuron_out][neuron_in]
            self.weighted_inputs.append(weighted_input)
            self.activations.append(self.activation(weighted_input))

        return self.activations
    
    def apply_gradients(self, learn_rate: float) -> None:
        for neuron_out in range(self.number_of_outputs):
            self.biases[neuron_out] -= learn_rate * self.loss_gradient_biases[neuron_out]
            for neuron_in in range(self.number_of_inputs):
                self.weights[neuron_out][neuron_in] -= learn_rate * self.loss_gradient_weights[neuron_out][neuron_in]
    

    # Helper methods.
    def initialize_parameters(self) -> None:
        # TODO: Implement various initialization algorithms such as LeCun,
        # Xavier, and He. Then, package them into their own classes.

        # Reset the previous parameters and their loss gradients.
        self.weights = []
        self.biases = []
        self.loss_gradient_weights = []
        self.loss_gradient_biases = []

        # Initialize them randomly.
        # NOTE: We are using LeCun Uniform initialization.
        for neuron_out in range(self.number_of_outputs):
            self.biases.append(0.0)
            self.loss_gradient_biases.append(0.0)
            # NOTE: Unlike weights, biases don't really need scaling, so values
            # like 0.0 for Sigmoid and Tanh-based networks and 0.01 for ReLU is fine.
            # NOTE: Using list comprehension for biases would surely be more
            # concise and performant, but we will refactor this initialization
            # method anyways.

            self.weights.append([])
            self.loss_gradient_weights.append([])
            for _neuron_in in range(self.number_of_inputs):
                self.loss_gradient_weights[neuron_out].append(0.0)

                random_value: float = random.uniform(-1.0, 1.0)
                self.weights[neuron_out].append(random_value / math.sqrt(self.number_of_inputs))

    def activation(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))