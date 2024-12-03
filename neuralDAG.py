import numpy as np
import logging
from typing import Callable, Dict, List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Activation functions and their derivatives
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0.0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU activation function."""
    return np.where(x > 0.0, 1.0, 0.0)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh activation function."""
    return 1.0 - np.tanh(x) ** 2

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid activation function."""
    return a * (1.0 - a)

def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function."""
    return x

def linear_derivative(x: np.ndarray) -> float:
    """Derivative of the linear activation function."""
    return 1.0

def xavier_init(fan_in: int, fan_out: int) -> float:
    """Xavier initialization for weights."""
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit)

class Edge:
    """
    Represents an edge (connection) between two nodes in the neural network graph.
    """
    def __init__(self, from_node: 'Node', to_node: 'Node', w: float):
        self.from_node = from_node
        self.to_node = to_node
        self.w = float(w)
        self.dw = 0.0

    def __repr__(self):
        return f"Edge(from={self.from_node.identifier}, to={self.to_node.identifier}, w={self.w})"

class Node:
    """
    Represents a node in the neural network graph.
    """
    def __init__(
        self,
        identifier: str,
        a: float = 0.0,
        b: float = 0.0,
        activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        activation_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        self.identifier = identifier
        self.a = float(a)
        self.b = float(b)
        self.edges: List[Edge] = []
        self.incoming_edges_count = 0
        self.incoming_edges: List[Edge] = []
        self.z = 0.0
        self.delta = 0.0
        self.label: Optional[float] = None
        self.db = 0.0
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def __repr__(self):
        return f"Node({self.identifier}, a={self.a}, b={self.b}, z={self.z})"

class DAG:
    """
    Represents a Directed Acyclic Graph for building custom neural network topologies.
    """
    def __init__(self, activation_function: str = 'tanh', cost_function: str = 'binary_cross_entropy'):
        self.nodes: Dict[str, Node] = {}

        activation_functions = {
            'tanh': (tanh, tanh_derivative),
            'relu': (relu, relu_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative),
            'linear': (linear, linear_derivative)
        }

        if activation_function in activation_functions:
            self.activation_function, self.activation_derivative = activation_functions[activation_function]
        else:
            raise ValueError("Unsupported activation function. Choose from 'relu', 'tanh', 'sigmoid', or 'linear'.")

        if cost_function == 'binary_cross_entropy':
            self.cost_function = self._binary_cross_entropy_cost
            self.cost_function_name = 'binary_cross_entropy'
        elif cost_function == 'mean_squared_error':
            self.cost_function = self._mean_squared_error_cost
            self.cost_function_name = 'mean_squared_error'
        else:
            raise ValueError("Unsupported cost function. Choose 'binary_cross_entropy' or 'mean_squared_error'.")

    def _binary_cross_entropy_cost(self, prediction: float, label: float) -> float:
        epsilon = 1e-7  # To avoid log(0)
        cost = - (label * np.log(prediction + epsilon) + (1.0 - label) * np.log(1.0 - prediction + epsilon))
        return cost

    def _mean_squared_error_cost(self, prediction: float, label: float) -> float:
        cost = 0.5 * (prediction - label) ** 2
        return cost

    def add_node(
        self,
        identifier: str,
        a: float = 0.0,
        b: float = 0.0,
        activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        activation_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        if identifier in self.nodes:
            raise ValueError(f"Node {identifier} already exists.")
        self.nodes[identifier] = Node(
            identifier, float(a), float(b),
            activation_function=activation_function,
            activation_derivative=activation_derivative
        )

    def add_edge(self, from_identifier: str, to_identifier: str):
        if from_identifier not in self.nodes or to_identifier not in self.nodes:
            raise ValueError("One or both nodes not found.")

        from_node = self.nodes[from_identifier]
        to_node = self.nodes[to_identifier]

        # Determine fan_in and fan_out for Xavier initialization
        fan_in = len(to_node.incoming_edges) if len(to_node.incoming_edges) > 0 else 1
        fan_out = len(from_node.edges) if len(from_node.edges) > 0 else 1

        w = xavier_init(fan_in, fan_out)

        edge = Edge(from_node, to_node, w)
        from_node.edges.append(edge)
        to_node.incoming_edges.append(edge)
        to_node.incoming_edges_count += 1

    def topological_sort(self) -> List[Node]:
        """
        Perform a topological sort of the nodes in the DAG.
        """
        incoming_edges_count = {node.identifier: node.incoming_edges_count for node in self.nodes.values()}
        visited = set()
        stack: List[Node] = []
        zero_incoming_nodes = [node for node in self.nodes.values() if incoming_edges_count[node.identifier] == 0]
        num_nodes = len(self.nodes)

        while zero_incoming_nodes:
            node = zero_incoming_nodes.pop()
            visited.add(node.identifier)
            stack.append(node)

            for edge in node.edges:
                to_node = edge.to_node
                incoming_edges_count[to_node.identifier] -= 1
                if incoming_edges_count[to_node.identifier] == 0:
                    zero_incoming_nodes.append(to_node)

        if len(visited) != num_nodes:
            raise ValueError("The graph has a cycle!")
        return stack

    def forward_pass(self):
        """
        Perform a forward pass through the network, computing activations.
        """
        topologically_sorted_nodes = self.topological_sort()
        for node in topologically_sorted_nodes:
            if len(node.incoming_edges) == 0:
                continue  # Input node; 'a' is already set
            node.z = sum(edge.from_node.a * edge.w for edge in node.incoming_edges) + node.b
            
            # Use node-specific activation function if provided
            activation_func = node.activation_function if node.activation_function is not None else self.activation_function
            node.a = activation_func(node.z)

    def backward_pass(self, output_nodes: List[Node]):
        """
        Perform a backward pass through the network, computing gradients.
        """
        topologically_sorted_nodes = self.topological_sort()

        for output_node in output_nodes:
            prediction = output_node.a
            label = output_node.label

            if self.cost_function_name == 'binary_cross_entropy':
                output_node.delta = prediction - label
            elif self.cost_function_name == 'mean_squared_error':
                # Use derivative of activation function at z
                activation_deriv = output_node.activation_derivative if output_node.activation_derivative else self.activation_derivative
                output_node.delta = (prediction - label) * activation_deriv(output_node.z)
            else:
                raise ValueError("Unsupported cost function.")

            output_node.db += output_node.delta  # Gradient for the bias of the output node

            # Compute gradients for incoming edges to the output node
            for edge in output_node.incoming_edges:
                edge.dw += edge.from_node.a * output_node.delta

        # Backpropagate to hidden layers
        for node in reversed(topologically_sorted_nodes):
            if node in output_nodes or len(node.incoming_edges) == 0:
                continue  # Skip the output nodes and input nodes

            # Compute the sum of weighted deltas from outgoing edges
            sum_w_delta = sum(edge.w * edge.to_node.delta for edge in node.edges)

            # Compute activation derivative
            activation_deriv = node.activation_derivative if node.activation_derivative else self.activation_derivative
            node.delta = activation_deriv(node.z) * sum_w_delta
            node.db += node.delta  # Accumulate gradient for mini-batch

            # Compute gradients for incoming edges
            for edge in node.incoming_edges:
                edge.dw += edge.from_node.a * node.delta  # Accumulate gradient for mini-batch

    def update_parameters(self, learning_rate: float, batch_size: int):
        """
        Update the parameters (weights and biases) of the network.
        """
        for node in self.nodes.values():
            node.b -= (learning_rate / batch_size) * node.db
            node.db = 0.0  # Reset accumulated gradient
        for node in self.nodes.values():
            for edge in node.edges:
                edge.w -= (learning_rate / batch_size) * edge.dw
                edge.dw = 0.0  # Reset accumulated gradient

    def get_input_node_ids(self) -> List[str]:
        """
        Get the identifiers of the input nodes.
        """
        input_nodes = [node.identifier for node in self.nodes.values() if len(node.incoming_edges) == 0]
        return input_nodes

    def _initialize_gradients(self):
        """
        Initialize (reset) the gradients before a new pass.
        """
        for node in self.nodes.values():
            node.db = 0.0
            node.delta = 0.0
        for node in self.nodes.values():
            for edge in node.edges:
                edge.dw = 0.0

    def _reset_activations(self):
        """
        Reset the activations and pre-activation values of the nodes.
        """
        for node in self.nodes.values():
            node.a = 0.0
            node.z = 0.0

    def _set_input_values(self, input_values: Union[List[float], np.ndarray], input_node_ids: List[str]):
        """
        Set the input values for the input nodes.
        """
        for idx, node_id in enumerate(input_node_ids):
            self.nodes[node_id].a = float(input_values[idx])

    def _compute_cost(self, prediction: float, label: float) -> float:
        """
        Compute the cost for a single prediction and label.
        """
        return self.cost_function(prediction, label)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 32,
        gradient_descent_type: str = 'mini-batch'
    ):
        """
        Train the neural network using the provided data.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.
            batch_size (int): Size of each mini-batch.
            gradient_descent_type (str): Type of gradient descent ('stochastic', 'mini-batch').
        """
        output_nodes = self.find_output_nodes()
        if not output_nodes:
            raise ValueError("No output nodes (nodes without outgoing edges) found!")

        m = X.shape[0]  # Number of training examples
        input_node_ids = self.get_input_node_ids()

        if gradient_descent_type == 'stochastic':
            batch_size = 1

        for epoch in range(epochs):

            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # For cost calculation per epoch 
            epoch_cost = 0.0

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self._initialize_gradients()

                batch_cost = 0.0

                for j in range(X_batch.shape[0]):
                    input_values = X_batch[j]
                    label = y_batch[j]

                    # Reset activations and z values
                    self._reset_activations()

                    # Set input values
                    self._set_input_values(input_values, input_node_ids)

                    # Set the labels for the output nodes
                    if hasattr(label, '__len__') and not isinstance(label, str):
                        if len(label) != len(output_nodes):
                            raise ValueError("Number of labels does not match number of output nodes.")
                        for idx, output_node in enumerate(output_nodes):
                            output_node.label = float(label[idx])
                    else:
                        if len(output_nodes) != 1:
                            raise ValueError("Expected a single label, but multiple output nodes found.")
                        output_nodes[0].label = float(label)

                    # Forward pass
                    self.forward_pass()

                    # Compute cost
                    predictions = [node.a for node in output_nodes]
                    labels = [node.label for node in output_nodes]
                    costs = [self._compute_cost(pred, lbl) for pred, lbl in zip(predictions, labels)]
                    cost = sum(costs)
                    batch_cost += cost

                    # Backward pass
                    self.backward_pass(output_nodes)

                # Update parameters after processing the batch
                self.update_parameters(learning_rate, X_batch.shape[0])

                epoch_cost += batch_cost

            # Compute average cost for the epoch
            average_cost = epoch_cost / m
            logger.info(f"Epoch {epoch + 1}/{epochs}, Cost: {average_cost}")

    def predict(self, input_values: Union[List[float], np.ndarray]) -> Union[float, List[float]]:
        """
        Make a prediction for the given input values.
        """
        self._reset_activations()

        input_node_ids = self.get_input_node_ids()

        if len(input_values) != len(input_node_ids):
            raise ValueError("Input values length does not match the number of input nodes.")
    
        self._set_input_values(input_values, input_node_ids)

        self.forward_pass()

        output_nodes = self.find_output_nodes()

        predictions = [node.a for node in output_nodes]

        if len(predictions) == 1:
            return predictions[0]
        else:
            return predictions

    def find_output_nodes(self) -> List[Node]:
        """
        Find and return the output nodes (nodes without outgoing edges).
        """
        output_nodes = [node for node in self.nodes.values() if len(node.edges) == 0]
        return output_nodes

    def validate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Validate the model on the provided data.
        """
        output_nodes = self.find_output_nodes()
        if not output_nodes:
            raise ValueError("No output nodes (nodes without outgoing edges) found!")

        m = X.shape[0]  # Number of examples
        input_node_ids = self.get_input_node_ids()

        predictions = []
        for i in range(m):
            input_values = X[i]
            pred = self.predict(input_values)
            predictions.append(pred)

        predictions = np.array(predictions)
        y_true = y

        if self.cost_function_name == 'binary_cross_entropy':
            # Classification problem
            y_pred = (predictions >= 0.5).astype(int)
            if predictions.ndim == 1:
                accuracy = np.mean(y_pred == y_true)
            else:
                accuracy = np.mean(np.all(y_pred == y_true, axis=1))
            logger.info(f"Validation Accuracy: {accuracy * 100:.2f}%")
            return accuracy
        elif self.cost_function_name == 'mean_squared_error':
            # Regression problem
            mse = np.mean((predictions - y_true) ** 2)
            logger.info(f"Validation MSE: {mse}")
            return mse
        else:
            raise ValueError("Unsupported cost function.")

    def visualize(self):
        """
        Visualize the DAG using networkx and matplotlib.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        # Add nodes with labels indicating their identifiers
        for node_id in self.nodes:
            G.add_node(node_id)

        # Add edges with weights
        for node in self.nodes.values():
            for edge in node.edges:
                G.add_edge(edge.from_node.identifier, edge.to_node.identifier)

        # Try to layout the graph in layers based on topological sort
        try:
            topological_order = list(nx.topological_sort(G))
            layer_dict = {}
            for node_id in topological_order:
                layer = len(nx.ancestors(G, node_id))
                layer_dict[node_id] = layer
            pos = nx.multipartite_layout(G, subset_key=lambda x: layer_dict[x])
        except:
            pos = nx.spectral_layout(G)

        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        plt.title('DAG Visualization')
        plt.axis('off')
        plt.show()

    def test(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Test the model on the provided data.
        """
        logger.info("Testing the model...")
        return self.validate(X, y)

    def __repr__(self):
        return f"DAG({', '.join([str(node) for node in self.nodes.values()])})"
