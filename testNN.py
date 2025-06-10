'''
input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

#computing the dot product of input vector and weights1 manually for practice
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult
print(f"The dot product is: {dot_product_1}")

#computing the dot product wiht the numpy library

dot_product_1 = np.dot(input_vector, weights_1)
dot_product_2 = np.dot(input_vector, weights_2)
#print(f"The dot product is: {dot_product_2}")
'''
import numpy as np
import matplotlib.pyplot as plt
'''
#hard coding the network iteratively
#wrapping the vectors in NumPy arrays
input_vector  = np.array([2, 1.5])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction = make_prediction(input_vector, weights_1, bias)

print(f"the prediction result is: {prediction}")

target = 0
mse = np.square(prediction - target)
print(f"Prediction: {prediction}; Error: {mse}")

derivative = 2*(prediction - target)
print(f"The derivative is {derivative}")

weights_1 = weights_1-derivative
prediction = make_prediction(input_vector, weights_1, bias)
error = (prediction - target) ** 2
print(f"Prediction: {prediction}; Error: {error}")
'''

#writing a class that does this
class NeuralNetwork:
    def __init__(self, learning_rate): #constructor
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def _sigmoid(self, x): #sigmoid function
        return 1/(1+np.exp(-x))
    
    def _sigmoid_deriv(self, x): #derivative of sigmoid function
        return self._sigmoid(x) * (1-self._sigmoid(x))
    
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction
    
    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias #check to see if I can just call the function above
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0* self.weights) + (1 * input_vector)

        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)

        return derror_dbias, derror_dweights
    
    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
    
    

#basic tests below without using train method
'''
if __name__ == "__main__":
    nn = NeuralNetwork(learning_rate = 0.1)
    input_vector = np.array([2, 1.5])
    target = 0

    for i in range(2):
        pred = nn.predict(input_vector)
        print(f"Step{i}: Prediction = {pred:.4f}")
        db, dw = nn._compute_gradients(input_vector, target)
        nn._update_parameters(db, dw)
        '''

#tests using the train mmethod

input_vectors = np.array([[3, 1.5], [2,1], [4, 1.5], [3,4], [3.5, 0.5], [2, 0.5], [5.5, 1], [1,1]])
targets = np.array([0,1,0,1,0,1,1,0])
learning_rate = 0.1

neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")


def test_model(model, inputs):
        for i,x in enumerate(inputs):
            pred = model.predict(x)
            print(f"Test input {i}: {x}, Prediction: {pred:.4f}")

new_inputs = np.array([ [3,2] , [1,0.5] , [4,4] ])
test_model(neural_network, new_inputs)