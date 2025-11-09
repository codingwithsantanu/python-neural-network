from .layer import Layer
from .training_data import TrainingData

class NeuralNetwork:
    EPSILON: float = 0.0001
    
    def __init__(self, *layer_sizes: int) -> None:
        self.layers: list[Layer] = []
        for layer in range(len(layer_sizes) - 1):
            self.layers.append(Layer(
                layer_sizes[layer],
                layer_sizes[layer + 1]
            ))
        # NOTE: The input layer is treated only as the raw inputs and not a
        # full-functional layer. We don't need to create an object for it.
    
    # Main methods.
    def feed_forward(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def classify(self, inputs: list[float]) -> int:
        predictions: list[float] = self.feed_forward(inputs)
        max_value_index: int = max(range(len(predictions)), key = predictions.__getitem__)
        return max_value_index
    
    def learn(self, batch: list[TrainingData], learn_rate: float) -> None:
        # TODO: Implement the mini-batching technique.
        # TODO: Implement back propagation.
        # TODO: Implement optimizers and momentum.
        
        # NOTE: We are using the finite-difference method.
        original_cost: float = self.calculate_total_cost(batch)

        for layer in self.layers:
            # Calculate loss gradients.
            for neuron_out in range(layer.number_of_outputs):
                # Calculate the change in cost for a small change in biases.
                layer.biases[neuron_out] += self.EPSILON
                delta_cost: float = self.calculate_total_cost(batch)
                layer.biases[neuron_out] -= self.EPSILON
                layer.loss_gradient_biases[neuron_out] = delta_cost / self.EPSILON

                # Calculate the change in cost for a small change in weights.
                for neuron_in in range(layer.number_of_inputs):
                    layer.weights[neuron_out][neuron_in] += self.EPSILON
                    delta_cost: float = self.calculate_total_cost(batch)
                    layer.weights[neuron_out][neuron_in] -= self.EPSILON
                    layer.loss_gradient_weights[neuron_out][neuron_in] = delta_cost / self.EPSILON

                # Apply the gradients.
                layer.apply_gradients(learn_rate)


    # Helper methods.
    def calculate_total_cost(self, batch: list[TrainingData]) -> float:
        total_cost: float = 0.0
        for data in batch:
            total_cost += self.calculate_cost(data)
        return total_cost
    
    def calculate_cost(self, data: TrainingData) -> float:
        predictions: list[float] = self.feed_forward(data.inputs)
        cost: float = 0.0
        for neuron_out in range(len(predictions)):
            cost += self.calculate_loss(predictions[neuron_out], data.expected[neuron_out])
        return cost

    def calculate_loss(self, output: float, expected: float) -> float:
        error: float = output - expected
        return error * error
        # NOTE: We are using the MSE loss function.
        # NOTE: We are squaring the loss or error to eliminate negatives and
        # emphasize larger errors which are more urgent than smaller errors.
        # TODO: Implement Cross Entropy.