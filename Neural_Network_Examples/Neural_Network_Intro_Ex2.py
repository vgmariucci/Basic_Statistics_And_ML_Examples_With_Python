##############################################################################################################################
#
# Now that we know the processes that occur in a neural network, we can create a class for it.
# Classes are the main building blocks of so-called object-oriented programming (OOP).
#
# The NeuralNetwork class will randomly generate initial values ​​for the weights and bias variables. In addition, when instantiating
# the NeuralNetwork object, we need to pass the alpha learning rate as a parameter. We will write a function to make predictions
# called prediction(), as well as methods to calculate the derivatives _calculates_derivatives() and to update the parameters
# __updates_parameters().
#  
################################################################################################################################
# Importing the libraries used in the example
import numpy as np
import matplotlib.pyplot as plt

print("\n================================================================================================================")
print("\n                                  CREATING THE NeuralNetwork CLASS                                      ")
print("\n================================================================================================================")

class NeuralNetwork:
    
    # Constructor method
    def __init__(self, alfa):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = alfa
    
   # Sigmoidal function method   
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Method that calculates the derivative of the sigmoidal function
    def __derivative_of_sigmoid(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    # Function to perform the prediction from the input vectors, weights and the bias value
    def prediction(self, v_input):
        layer_1 = np.dot(v_input, self.weights) + self.bias
        layer_2 = self.__sigmoid(layer_1)
        prediction = layer_2
        return prediction
    
    # Function to perform the prediction from the input vectors, weights and the bias value 
    def __calculate_gradient(self, v_input, v_correct_answers):
        layer_1 = np.dot(v_input, self.weights) + self.bias
        layer_2 = self.__sigmoid(layer_1)
        prediction = layer_2
        
        derivative_prediction_error_layer_2 = 2 * (prediction - v_correct_answers)
        
        derivative_prediction_layer_1 = self.__derivative_of_sigmoid(layer_1)
        
        derivative_bias_layer_1 = 1
        
        derivative_weights_layer_1 = (0 * self.weights) + (1 * v_input)
        
        bias_adjustment_value = derivative_prediction_error_layer_2 * derivative_prediction_layer_1 * derivative_bias_layer_1
        
        adjustment_value_of_the_weights = derivative_prediction_error_layer_2 * derivative_prediction_layer_1 * derivative_weights_layer_1
        
        return bias_adjustment_value, adjustment_value_of_the_weights
    
    
    def __update_parameters(self, bias_adjustment_value, adjustment_value_of_the_weights):
        
        self.bias = self.bias - (bias_adjustment_value * self.learning_rate)
        
        self.weights = self.weights - (adjustment_value_of_the_weights * self.learning_rate)
        
    
    def training(self, v_inputs, v_correct_answers, number_of_iterations):
        
        cumulative_errors = []
        
        for i in range(number_of_iterations):
            # Randomly selects a training data
            random_index = np.random.randint(len(v_inputs))
            
            vetor_input = v_inputs[random_index]
            correct_answer_vector = v_correct_answers[random_index]
        
            # Calculate the gradient and update the weights e o bias
            bias_adjustment_value, adjustment_value_of_the_weights = self.__calculate_gradient(vetor_input, correct_answer_vector)
        
            self.__update_parameters(bias_adjustment_value, adjustment_value_of_the_weights)
        
            # If the iteration is a multiple of 100, we execute the following lines of code to see how the error behaves every 100 iterations
            if i % 100 == 0:
                
                cumulative_error = 0
            
                # Loop through all error values ​​generated in each iteration
                for data_index in range(len(v_inputs)):
                    
                    dado_de_input = v_inputs[data_index]
                    correct_answer = v_correct_answers[data_index]
                    
                    prediction = self.prediction(dado_de_input)
                    erro = np.square(prediction - correct_answer)
                    
                    cumulative_error = cumulative_error + erro
                    
                cumulative_errors.append(cumulative_error)
        
        return cumulative_errors  
        
##############################################################################################################################
#
# Once our NeuralNetwork class is created, simply create an instance of it and call the .prediction() function so that
# the network can make a prediction for the output responses based on the input variables
#
##############################################################################################################################

# Before instantiating our Neural Network, we first choose the value of the learning rate alpha
alfa = 0.1

# Instantiating our Neural Network
rede_neural = NeuralNetwork(alfa)

# Declaring the input vectors:
v_input = np.array([1.5 , 2])

predicted_value = rede_neural.prediction(v_input)

print(f"\n The value predicted by the neural network was: {predicted_value}")

################################################################################################################################
#
# Although our neural network is making predictions, we still need to train it. The goal is to make the network learn to
# detect patterns of correct responses based on the input data. This means that the neural network needs to know how to
# adapt to new input data that has the same probability distribution as the training data.
#
# Stochastic Gradient Descent is a technique in which, at each iteration, the model (neural network) makes a prediction
# for randomly selected training data, that is, in a stochastic manner, calculating the error and updating the
# parameters.
#
# To do this, we need to create a method for the NeuralNetwork class that will train our neural network with training data.
#
# We will also save the errors of each iteration in order to show through a graph how the error behaves
# as the number of iterations increases.
###############################################################################################################################
print("\n================================================================================================================")
print("\n                                  TRAINING THE NEURAL NETWORK WITH MORE DATA                                     ")
print("\n================================================================================================================")

# Dataset 1
# Input values
v_input = np.array(
                    [
                        [3, 1.5], 
                        [2, 1], 
                        [4, 1.5], 
                        [3, 4], 
                        [3.5, 0.5], 
                        [2, 0.5], 
                        [5.5, 1], 
                        [1, 1]
                    ]
                   )

# Output values
v_correct_answers = np.array([0, 1, 0, 1, 0, 1, 1, 0])

# Dataset 2
# Input values
# v_input = np.array(
#                     [
#                         [3, 1.5], 
#                         [2, 1], 
#                         [4, 1.5], 
#                         [3, 4], 
#                         [3.5, 0.5], 
#                         [2, 0.5], 
#                         [5.5, 1], 
#                         [1, 1],
#                         [4, 2.5], 
#                         [2.5, 1], 
#                         [2.5, 1.5], 
#                         [1, 0.7], 
#                         [1, 1], 
#                         [2.2, 0.5], 
#                         [3.7, 2], 
#                         [3, 3]
#                     ]
#                    )

# Output values
# v_correct_answers = np.array([0, 1, 0, 1, 0, 1, 1, 0,
#                                  1, 0, 0, 0, 0, 0, 1, 1])

number_of_iterations = 10000

training_error = rede_neural.training(v_input, v_correct_answers, number_of_iterations )

plt.plot(training_error)
plt.xlabel("Iterações")
plt.ylabel("Erro durante as iterações")
plt.show()

##########################################################################################################
# 
# From the Error vs. Iteration graph, we can see that the total error starts with a high value and tends to a relatively lower value and oscillates around an average value.
# These sudden oscillations are due to the random selection of training data, as well as the small amount of data.
#
# It is not recommended to use training data to evaluate the performance of the neural network, since this data is data that, after training, it will already know how to respond to the inputs. This can lead to overfitting, when the neural network becomes so good at predicting the training data that it is unable to generalize to new data.
#
# In this example, the main objective is to understand the basic foundations of building a neural network, which is why we use a small set of data. Deep learning models generally need a large amount of data due to the complexities of certain problems, such as image recognition or audio signals, among others. Due to the different levels of complexity,
# using only one or two layers in the neural network is not enough, so we call it deep learning
# precisely because the neural network is composed of many layers.
#
# By adding more layers to the neural network and various types of activation functions, we increase its
# prediction power. An example of an application of this level of complexity is facial recognition, as
# some cell phones have when unlocking the screen when they recognize the owner or who has been
# registered to use it.
############################################################################################################
