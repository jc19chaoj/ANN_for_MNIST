import numpy as np

def relu(x):
    return np.maximum(x, 0).astype(np.float32)

def relu_derive(x):
    return np.greater(x, 0).astype(np.float32)

class ANN:
    def __init__(self, ANN_size, inputs, labels):
        # ANN_size is a list of the number of neurons in each layer.
        # For example, ANN_size = [784, 256, 256, 10]
        print("ANN size is:", ANN_size)
        self.input_size = ANN_size[0] # The input size would be 784
        self.hidden_size = ANN_size[1:-1] if isinstance(ANN_size[1:-1], list) else [ANN_size[1:-1]] # The hidden size would be [256, 256].
        self.output_size = ANN_size[-1] # The output size would be 10

        # inputs and labels
        self.inputs = inputs
        self.labels = labels

        # weights
        self.hidden_weights = []
        self.hidden_weights.append(np.random.rand(self.input_size, self.hidden_size[0]))

        # If more than one hidden layer
        for i in range(1, len(self.hidden_size)):
            self.hidden_weights.append(np.random.rand(self.hidden_size[i-1], self.hidden_size[i]))

        self.output_weights = np.random.rand(self.hidden_size[-1], self.output_size)

        # biases
        self.hidden_biases = []
        self.hidden_biases.append(np.random.rand(1,self.hidden_size[0]))

        # If more than one hidden layer
        for i in range(1, len(self.hidden_size)):
            self.hidden_biases.append(np.random.rand(1, self.hidden_size[i]))

    def forward(self):
        s_hidden = []
        s_hidden.append(np.matmul(self.inputs, self.hidden_weights[0]))

        hidden = []
        hidden.append(relu(s_hidden[0]))

        # If more than one hidden layer
        for i in range(1, len(self.hidden_size)):
            s_hidden.append(np.matmul(hidden[i-1], self.hidden_weights[i]))
            hidden.append(relu(s_hidden[i]))

        s_output = np.matmul(hidden[-1], self.output_weights)
        self.output = s_output # No activation function needed

        # prediction accuracy
        self.prediction = np.equal(np.argmax(self.output),np.argmax(self.labels))


def main():
    inputs = np.random.rand(28, 28)
    labels = np.random.rand(1, 10)
    inputs_reshape = np.reshape(inputs, (1, 784))

    ANN1 = ANN([784,256,256,10], inputs_reshape, labels)
    print(ANN1.input_size, ANN1.hidden_size, ANN1.output_size)
    for i in range(len(ANN1.hidden_size)):
        print("hidden", i+1, "weights:", ANN1.hidden_weights[i].shape)
    print("output weights:", ANN1.output_weights.shape)

    ANN1.forward()
    print("Predict:", ANN1.prediction)


if __name__ == "__main__":
    main()
