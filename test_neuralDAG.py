

def test():
    import numpy as np
    import logging
    from neuralDAG import DAG, linear, linear_derivative

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    np.random.seed(42)
    m = 1000  # Number of samples
    X = np.random.randn(m, 3)  # 3 inputs

    y1 = np.sin(X[:, 0]) + np.cos(X[:, 1])  # Output 1
    y2 = X[:, 0] * X[:, 1] + X[:, 2]       # Output 2
    y = np.column_stack((y1, y2))           # Combine outputs

    dag1 = DAG(activation_function='tanh', cost_function='mean_squared_error')

    dag1.add_node('Input1', activation_function=None, activation_derivative=None)
    dag1.add_node('Input2', activation_function=None, activation_derivative=None)
    dag1.add_node('Input3', activation_function=None, activation_derivative=None)

    dag1.add_node('Hidden1')
    dag1.add_node('Hidden2')
    dag1.add_node('Hidden3')

    dag1.add_node('Output1', activation_function=linear, activation_derivative=linear_derivative)
    dag1.add_node('Output2', activation_function=linear, activation_derivative=linear_derivative)

    dag1.add_edge('Input1', 'Hidden1')
    dag1.add_edge('Input2', 'Hidden1')
    dag1.add_edge('Input2', 'Hidden2')
    dag1.add_edge('Input3', 'Hidden2')
    dag1.add_edge('Input1', 'Hidden3')
    dag1.add_edge('Input3', 'Hidden3')

    dag1.add_edge('Hidden1', 'Output1')
    dag1.add_edge('Hidden2', 'Output1')
    dag1.add_edge('Hidden2', 'Output2')
    dag1.add_edge('Hidden3', 'Output2')

    dag1.add_edge('Input1', 'Output1')
    dag1.add_edge('Input2', 'Output2')

    dag1.train(X, y, epochs=50, learning_rate=0.01, batch_size=32)

    dag1.validate(X, y)

    test_input = np.array([0.5, -1.2, 0.3])
    prediction = dag1.predict(test_input)
    print("Test Input:", test_input)
    print("Predicted Outputs:", prediction)

    dag1.visualize()

def test2():
    import numpy as np
    from neuralDAG import DAG, linear, linear_derivative

    np.random.seed(42)
    m = 1000  # Number of samples
    X = np.random.randn(m, 4)  # 4 inputs

    y1 = X[:, 0] + X[:, 1]**2 - X[:, 2]
    y2 = np.sin(X[:, 2]) + np.cos(X[:, 3])
    y = np.column_stack((y1, y2))

    dag2 = DAG(activation_function='relu', cost_function='mean_squared_error')

    dag2.add_node('Input1', activation_function=None, activation_derivative=None)
    dag2.add_node('Input2', activation_function=None, activation_derivative=None)
    dag2.add_node('Input3', activation_function=None, activation_derivative=None)
    dag2.add_node('Input4', activation_function=None, activation_derivative=None)

    dag2.add_node('Hidden1')
    dag2.add_node('Hidden2')
    dag2.add_node('Hidden3')

    dag2.add_node('Output1', activation_function=linear, activation_derivative=linear_derivative)
    dag2.add_node('Output2', activation_function=linear, activation_derivative=linear_derivative)

    dag2.add_edge('Input1', 'Hidden1')
    dag2.add_edge('Input2', 'Hidden1')
    dag2.add_edge('Input3', 'Hidden2')
    dag2.add_edge('Input4', 'Hidden2')
    dag2.add_edge('Hidden1', 'Hidden3')
    dag2.add_edge('Input2', 'Hidden3')  # Skip connection from Input2 to Hidden3

    dag2.add_edge('Hidden3', 'Output1')
    dag2.add_edge('Hidden2', 'Output2')

    dag2.add_edge('Input1', 'Output1')
    dag2.add_edge('Input3', 'Output2')

    dag2.train(X, y, epochs=50, learning_rate=0.01, batch_size=32)

    dag2.validate(X, y)

    test_input = np.array([0.1, 0.2, -0.3, 0.4])
    prediction = dag2.predict(test_input)
    print("Test Input:", test_input)
    print("Predicted Outputs:", prediction)

    dag2.visualize()

def test3():
    import numpy as np
    from neuralDAG import DAG, linear, linear_derivative

    np.random.seed(42)
    m = 1000  # Number of samples
    X = np.random.randn(m, 2)  # 2 inputs

    y1 = np.log(np.abs(X[:, 0]) + 1) * X[:, 1]
    y2 = np.exp(-X[:, 0] * X[:, 1])
    y = np.column_stack((y1, y2))

    dag3 = DAG(activation_function='tanh', cost_function='mean_squared_error')

    dag3.add_node('Input1', activation_function=None, activation_derivative=None)
    dag3.add_node('Input2', activation_function=None, activation_derivative=None)

    dag3.add_node('Hidden1')
    dag3.add_node('Hidden2')

    dag3.add_node('Output1', activation_function=linear, activation_derivative=linear_derivative)
    dag3.add_node('Output2', activation_function=linear, activation_derivative=linear_derivative)

    dag3.add_edge('Input1', 'Hidden1')
    dag3.add_edge('Input2', 'Hidden1')
    dag3.add_edge('Input1', 'Hidden2')
    dag3.add_edge('Hidden1', 'Hidden2')  # Non-feedforward connection

    dag3.add_edge('Hidden2', 'Output1')
    dag3.add_edge('Hidden1', 'Output2')

    dag3.add_edge('Input2', 'Output1')

    dag3.train(X, y, epochs=50, learning_rate=0.01, batch_size=32)

    dag3.validate(X, y)

    test_input = np.array([0.5, -0.5])
    prediction = dag3.predict(test_input)
    print("Test Input:", test_input)
    print("Predicted Outputs:", prediction)

    dag3.visualize()

def test4():
    import numpy as np
    from neuralDAG import DAG, linear, linear_derivative
    np.random.seed(42)
    m = 1000  # Number of samples
    X = np.random.randn(m, 3)  # 3 inputs

    # Outputs
    y1 = X[:, 0] ** 2 + X[:, 1] ** 2
    y2 = X[:, 1] * X[:, 2] + np.sin(X[:, 0])
    y = np.column_stack((y1, y2))

    dag4 = DAG(activation_function='relu', cost_function='mean_squared_error')

    dag4.add_node('Input1', activation_function=None, activation_derivative=None)
    dag4.add_node('Input2', activation_function=None, activation_derivative=None)
    dag4.add_node('Input3', activation_function=None, activation_derivative=None)

    dag4.add_node('Hidden1')
    dag4.add_node('Hidden2')
    dag4.add_node('Hidden3')

    dag4.add_node('Output1', activation_function=linear, activation_derivative=linear_derivative)
    dag4.add_node('Output2', activation_function=linear, activation_derivative=linear_derivative)

    dag4.add_edge('Input1', 'Hidden1')
    dag4.add_edge('Input2', 'Hidden1')
    dag4.add_edge('Input2', 'Hidden2')
    dag4.add_edge('Input3', 'Hidden2')
    dag4.add_edge('Input1', 'Hidden3')
    dag4.add_edge('Input3', 'Hidden3')

    dag4.add_edge('Hidden1', 'Output1')
    dag4.add_edge('Hidden2', 'Output1')
    dag4.add_edge('Hidden2', 'Output2')
    dag4.add_edge('Hidden3', 'Output2')

    # Additional connections to create multiple paths
    dag4.add_edge('Input1', 'Output1')
    dag4.add_edge('Input2', 'Output1')
    dag4.add_edge('Hidden1', 'Output2')
    dag4.add_edge('Hidden3', 'Output1')

    dag4.train(X, y, epochs=50, learning_rate=0.01, batch_size=32)

    test_input = np.array([1.0, 2.0, -1.0])
    prediction = dag4.predict(test_input)
    print("Test Input:", test_input)
    print("Predicted Outputs:", prediction)

    dag4.visualize()


if __name__ == '__main__':
    test()
    test2()
    test3()
    test4()