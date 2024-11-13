# ###############################################################################################################################

# CONSTRUCTION OF OUR FIRST NEURAL NETWORK

# ###############################################################################################################################

# The first step in building our neural network is to produce an output response from some input information. 
# We can do this by creating a weighted sum of the input information. 
# In this case, we will use the NumPy library to represent the input information.

# ################################################################################################################################
 
# REPRESENTING NEURAL NETWORK INPUT INFORMATION WITH NumPy

# ################################################################################################################################                                

# We will use the NumPy library to represent the input information (vectors) in the form of
# arrays. However, before we use NumPy, it is interesting to use just a few features
# of the Python language itself to better understand how some parts of the process in neural networks
# work.

# In this first example, we have an input vector v_in and two other vectors v_1 and v_2.
# The goal is to find which vector (v_1 or v_2) is most similar to the input vector v_in, taking into account 
# the direction and magnitude of the vectors.

# Q: How can we compare vectors using Python?

# A:
# - First, let's define the 3 vectors (v_in, v_1 ​​and v_2);
# - Then we calculate how similar v_in is to v_1 and v_2 through the scalar products,
# also known in linear algebra as the inner product (dot_product) between the vectors v_in and v_1
# and v_in and v_2. 
# The highest resulting value between the scalar products is associated with the highest
# similarity between the vectors.

# ###############################################################################################################################

# Importing the libraries used in the example
import numpy as np
import matplotlib.pyplot as plt

print("\n================================================================================================================")
print("\n                                    INTRODUCTION: REVIEW ON SCALAR PRODUCT                                   ")
print("\n================================================================================================================")

# Input vectors
v_in = [2, 3]
v_1  = [2, 0]
v_2  = [2, 4]


######################################################################################################
#
# Calculate the dot product using Python commands manually (element by element of each vector)
#
########################################################################################################

# multiplication_first_indices = v_in[0] * v_1[0]
# multiplicacao_seconds_inidices  = v_in[1] * v_1[1]
# dot_product_v_in_v_1 = multiplication_first_indices + multiplicacao_seconds_inidices

# multiplication_first_indices = v_in[0] * v_2[0]
# multiplicacao_seconds_inidices  = v_in[1] * v_2[1]
# dot_product_v_in_v_2 = multiplication_first_indices + multiplicacao_seconds_inidices

# print("\n The dot product between v_in e v_1 is equal: ", dot_product_v_in_v_1)
# print("\n The dot product between v_in e v_2 is equal: ", dot_product_v_in_v_2)

##########################################################################################################

##########################################################################################################
#
# Calculate the dot product between two vectors by defining a function in Python
#
###########################################################################################################
def calculate_dot_product(u, v):
 
 dot_product = 0
 
 for i in range(0, len(u)):
     
    dot_product += u[i] * v[i]
    
 return dot_product

print("\n The dot product between v_in = [2, 3] e v_1 = [2, 0] is equal: ", calculate_dot_product(v_in, v_1))

print("\n The dot product between v_in = [2, 3] e v_2 = [2, 4] is equal: ", calculate_dot_product(v_in, v_2))

###########################################################################################################
#
# Calculate the dot product between two vectors using NumPy
#
###########################################################################################################

print("\n The dot product between v_in = [2, 3] e v_1 = [2, 0] with NumPy: ", np.dot(v_in, v_1))

print("\n The dot product between v_in = [2, 3] e v_2 = [2, 4] with NumPy: ", np.dot(v_in, v_2))

###########################################################################################################
#
# Comparison of the results between the scalar products, indicating the similarity between the vectors
#
###########################################################################################################

if  (np.dot(v_in, v_1) > np.dot(v_in, v_2)):
    print("\n The vector v_1 is most similar to the vector v_in")
elif(np.dot(v_in, v_1) < np.dot(v_in, v_2)) :
    print("\n The vector v_2 is most similar to the vector v_in")
else:
    print("\n v_1 and v_2 have the same degree of similarity to v_in")
    
############################################################################################################


############################################################################################################
#
# In this example we will train a model to make predictions so that the output answers can be
# only 0 or 1. This is a classification problem, a subset of supervised machine learning methods,
# in which we have a set of input data, as well as the answers for the outputs:
#
#                                   Input Vector                   Output Response
#                                      [1.5, 2]                                1
#                                      [2, 3.5]                                0
#
# 
# In this example, we are working with a data set composed simply of numbers.
# In many applications, the data set is quite different, for example, it can be in the form of a
# text file, a table with values, an audio signal, a photo or a video.
#
# To make our example simple, we will develop a neural network composed of only 2 layers.
# We have already seen that the only mathematical operations performed in the neural network are the scalar product and a sum,
# both of which are linear operations.
#
# If we add more layers and at the same time maintain only these two linear operations, the final result
# will reveal that no effect different from what we have already seen is detected, since each layer will always have
# some correlation with the input of the previous layer. This implies that, for a neural network with multiple
# layers, there will always be a neural network with fewer layers, that is, a simpler one, that will be able to predict
# the same result.
#
# We need to find a way (a mathematical operation) that makes the layer(s) correlate only
# sometimes. In other words, we need to add a non-linearity feature.
#
# We can do this using specific non-linear functions known as activation functions.
# There are several types of activation functions, one of the most widely used nowadays 
# is the ReLU (Rectified Linear Unit) activation function.   
#  
#  ReLU has the following form:
#
#   f(x) = x , x > 0
#   f(x) = 0 , x <= 0                            
#                                                f(x)          
#                                                |          . 
#                                                |       .  
#                                                |    .    
#                                                | .     
#                                  -------------O: -------------->x 
#                                                | 
#                                                |
#                                                |
#  
# In short, the ReLU activation function transforms every negative x input into 0 and every positive x input
# into an equivalent value equal to the value of x (a straight line with a 45 degree slope or tan(45 degrees) = 1).
#
# However, in this example we will adopt another activation function known as the sigmoidal function, given by:
#
#   f(x) = 1 / (1 + exp[-x])
#                                               f(x)          
#                                               1|   x x x x x   
#                                                | x       
#                                            0.5 x        
#                                              x |      
#                                  -x-x-x-x-x---O: -------------->x 
#                                                | 
#                                                |
#                                                |
#  
# The only possible responses in our data set are 0 and 1, so the sigmoidal function is
# able to limit the neural network's response to the range of 0 and 1.
#
# Probability functions give us the odds of the possible responses or events occurring.
# If the output is greater than 0.5 (50%), then we can assume that the prediction will be 1 or 100%.
# If the output is less than 0.5 (50%), then we assume that the prediction will be 0 or 0%.
#
############################################################################################################

print("\n================================================================================================================")
print("\n       CONSTRUCTION OF OUR NEURAL NETWORK WITH 2 LAYERS (1 INPUT LAYER AND 1 OUTPUT LAYER)                      ")
print("\n================================================================================================================")

# Declaring the input vectors, weights, bias and the correct answer:
v_entry = np.array([1.5 , 2])
v_pesos = np.array([2, 1.2])
bias = np.array([0.0])

# Declaring the correct answer for comparison when v_entry = np.array([1.5 , 2])
correct_answer = 1; 

# Definition of the sigmoidal function
def sigmoid(x):
    
    sigmoidal_of_x = 1 / (1 + np.exp(-x))
    
    return sigmoidal_of_x 

def perform_prediction(v_entry, v_pesos, bias):
    
    camada_1 = np.dot(v_entry, v_pesos) + bias
    
    camada_2 = sigmoid(camada_1)

    return camada_2

prediction = perform_prediction(v_entry, v_pesos, bias)

print(f"\n The correct answer is: {correct_answer}")
print(f"\nThe predicted value is:    {prediction}")

#############################################################################################################
#
# The prediction result is 0.9889 (~99%), or practically 100%. Therefore, the neural network made a
# correct prediction. Let's try to make a prediction with the other input vector from the previous table, that is:
#
# v_entry = [2 , 3.5]
#
# We already know that the correct answer for this input is 0.
############################################################################################################

v_entry = np.array([2, 3.5])

# Correct answer quando v_entry = np.array([2 , 3.5])
correct_answer = 0 

prediction = perform_prediction(v_entry, v_pesos, bias)

print(f"\n The correct answer is: {correct_answer}")
print(f"\nThe predicted value is:    {prediction}")

############################################################################################################
#
# This time the result was 0.99 (practically 1) instead of 0. Therefore, the neural network made a wrong
# prediction.
#
# But how wrong is this prediction?

# Let's try to calculate the size of this error?
#
############################################################################################################

############################################################################################################
#  
# In the neural network training process, the first step is to calculate the error of each prediction
# and then adjust the weights proportionally to the error value. To do this, we use a calculation tool
# called "gradient descent" and a procedure known as "backpropagation".
# 
# Gradient descent is used to obtain the "direction" (sign, positive or negative) and the rate at which
# the weights should be adjusted based on the errors.
#
# To understand the magnitude of the error, we need a method to measure it. 
# The function to calculate the error is called the "cost function" or "loss function". 
# In this example, we will use the mean squared error (MSE) as the cost function. 
# 
# To do this, we first calculate the difference between the value predicted by the neural network and the 
# real value (which is the output response that the neural network should predict), then we square the 
# result of this difference. 
# The neural network can make mistakes above or below the real value, which will generate positive or 
# negative values. 
# 
# Since we chose our cost function to be the mean squared error, we always end up with positive values.
############################################################################################################
print("\n==================================================================================================")
print("\n                               CALCULATING PREDICTION ERROR                                     ")
print("\n==================================================================================================")

correct_answer = 0

mse = np.square(prediction - correct_answer)

print(f"\n Prediction:{prediction}; MSE:{mse}")

#################################################################################################################
#
# We can see that the value of the cost function is equal to the prediction value, being mse = 0.995...
# since the correct answer is 0.
#
# One implication of squaring the difference between the prediction and the correct answer is that large differences 
# will generate even larger errors, while small differences will generate smaller and smaller errors that do not have 
# a significant impact on the input weight correction process (in the case where the errors are << 1). 
# The goal is to vary the values ​​of the weights and biases in order to reduce the error. 
# To understand how this works, we will only change the values ​​of the weights without changing the value of the bias. 
# In addition, we can leave out the second layer that uses the sigmoidal function. Thus, the cost function can be 
# represented as a quadratic equation (parabola) centered at the origin:
#
#   mse = (prediction - resposta_correta)^2  -->  y = E^2, com E = (prediction - resposta_correta) e y = mse 
#                                                 
#                                                 y = mse   
#                                        .       |        .
#                                         .      |       .
#                                          ª     |      º 
#                                            .   |    .  
#                                  ------------ O:--------------->E 
#                                                | 
#                                                |
# 
# For example, if the error E is equivalent to the value of the coordinate of point ª, then we have a negative rate of change
# (slope of the curve -> derivative -> gradient descent), so we need to increase the predicted value
# so that the MSE tends to zero, as well as its rate of change. The other case is when the error is equivalent
# to the value of the coordinate of point º. In this case, the gradient descent is positive, so it is necessary to decrease
# the predicted value so that MSE is reduced to zero.
#
# Therefore, gradient descent allows us to obtain two important characteristics related to the prediction error of our neural network: 
# 1- Error rate or proportion; 
# 2- Error sign ("direction"), whether it is positive (prediction > correct_answer) or negative (prediction < correct_answer)

# Knowing these two characteristics related to the prediction error, we can adjust the values ​​of the input # people so that the 
# prediction is increasingly closer to the correct answer, minimizing the error.
# 
# When we have prediction = correct_answer, then mse = 0.
#
# Let's try to adjust the error using the derivative (gradient descent) of our cost function y = mse.
#
# In this case, we have that the derivative of y = E^2 is y' = 2E
#
# Remembering that E = (prediction - correct_answer).
#
#################################################################################################################
print("\n================================================================================================================")
print("\n                      ADJUSTING INPUT WEIGHTS ACCORDING TO NEURAL NETWORK ERROR                           ")
print("\n================================================================================================================")

derivada_mse = 2 * (prediction - correct_answer)

print(f"\n The derivative (gradient descent) for the cost function (MSE) is: {derivada_mse}")

#####################################################################################################################
# 
# The result of the derivative is 1.9994, that is, it is a positive value and approximately equal to 2, since
# the cost function is 0.9994 ~ 1.
#
# This means that we need to reduce the weight values. This is done by subtracting from the values ​​of the weights
# declared at the input of the neural network the value generated by the derivative of the cost function.
#
####################################################################################################################

v_adjusted_weights = v_pesos - derivada_mse

prediction = perform_prediction(v_entry, v_adjusted_weights, bias)

mse = np.square(prediction - correct_answer)

print(f"\n Adjusted Weights: {v_adjusted_weights}")

print(f"\n Prediction:{prediction}; MSE:{mse}")

####################################################################################################################
#
# We can see that the cost function has decreased drastically (previous mse = 0.99 and current mse = 0.003),
# so that the neural network makes a correct prediction this time (predicted value = 0.005 ~ 0 and correct answer = 0).
#
# In this example the value of the derivative was small, however, there are situations in which the value of the derivative can be
# large (in modulus). In this situation, the adjustment of the input weights is directly proportional to the value of the derivative
# (or gradient descent) becoming excessive and generating even larger errors.
#
# One way to illustrate such an extreme situation would be to obtain a high value for the derivative at point º of the graph
# of the parabola. By adjusting the weights with such a high value, we could obtain a large error again,
# for example, near point ª (on the opposite side of the parabola curve). In this way, the neural network would be
# switching from one point to another and would never reduce the error between predictions. To solve this problem,
# we adjust the weights using only a fraction of the value of the derivative of the cost function.
#
# We use a variable to define what fraction of the derivative value we will consider to adjust the weights.
# This variable is called the learning rate and we can represent it by the letter alpha
# (called the alpha parameter). If we decrease the learning rate, then the adjustments to the input weights
# will be smaller, the opposite occurs when we increase the learning rate, that is, the adjustments to the weights will be
# by larger values.
#
# How can we know what the best learning rate is?
#
# The answer is: By trial and error!
#
# In conventional artificial neural networks, values ​​such as:
#
# alfa -> 0.1, 0.01 e 0.001 
#
# If our neural network makes correct predictions for each input value during training, then we probably have an overfit model, 
# in which the neural network simply memorizes how to classify the input values ​​instead of learning to detect patterns or 
# characteristics in them. There are some techniques to overcome this behavior of the neural network, such as "regularization methods"
# and "stochastic gradient descent", the latter of which uses randomly generated values. 
# In this example, we will use stochastic gradient descent as a tool to avoid overfitting.
#
####################################################################################################################
#
# In our neural network, we need to adjust both the weights and the bias. Therefore, the cost function that we will use to
# measure the error depends on two independent variables.
#
# We want to know how to adjust the weights and the bias so that the error is reduced.
#
# Since we have this composition of functions, when performing the derivative of the cost function, we need to use
# the "chain rule" of derivation techniques. With the chain rule, we partially differentiate each function,
# calculating for each one the value when E = (prediction - correct_answer). The final result of this process
# is given by the multiplication of all the values ​​of each partial derivative calculated for x. This final result
# will be the value with which the input weights will be adjusted and we will call it the _adjustment_value_of_the_weights

#  weights_adjustment_value = derivative_prediction_error_layer_2 * derivative_prediction_layer_1 * derivative_weights_layer_1
#                                                          
#
# Remember that layer 2 is the one that performs the final prediction of our neural network.
#
# Now we can start adjusting the weights in a process known as Backpropagation.
#
# As the word itself says, backpropagation performs the differentiation process using the chain rule
# from the final layer to the initial layer (backwards). The final layer in our neural network is the
# second layer containing the sigmoidal function f(x) = 1 / (1 + exp(-x))
# 
# The derivative of the sigmoidal function is given by: f'(x) = f(x)(1-f(x))
#
# When performing the derivative of layer 1 with respect to the bias, we have:
#
#  bias_adjustment_value = derivative_prediction_error_layer_2 * derivative_prediction_layer_1 * derivative_bias_layer_1
#                              
#  However, we have that derivative_bias_layer_1 = 1                         
#
#
#######################################################################################################################################