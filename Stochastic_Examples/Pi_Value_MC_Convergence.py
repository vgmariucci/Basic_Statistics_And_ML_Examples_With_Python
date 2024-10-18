##########################################################################################################
# 
# This script allows us to visualize the speed of convergence of the Monte Carlo method for the example
# of estimating the value of pi, as well as understanding the dependence of the error (estimated value - true value)
# as a function of the number of iterations N of the Monte Carlo method.
#
########################################################################################################
import numpy as np
import random
import matplotlib.pyplot as plt


# Number of iteractions
N = [10, 100, 1000]

for number_of_interactions in N:

    # Resets ("clears") the droplet count in the circle area before starting the simulation
    n_in = 0

    estimated_value = np.empty(number_of_interactions)

    # Distribute the raindrops (random points) over the area of ​​the square:
    for i in range(number_of_interactions):
    
        x = random.random()
        y = random.random()
        
        # If a drop fell inside the circle, then count that drop and show its coordinate
        if np.sqrt((x**2) + (y**2)) <= 1:
            n_in += 1
            estimated_value[i] = 4 * n_in /float(i + 1)
    
    print(estimated_value)
    
    plt.xlabel("Number of Iterations")
    # Displays the exact value of pi with a horizontal line on the graph
    plt.axhline(np.pi, c = 'r', label = "Pi Value") 
    plt.plot(1/np.sqrt(np.arange(number_of_interactions)+1), '--g', label = "Estimated Error")
    plt.plot(np.arange(number_of_interactions), np.abs(estimated_value-np.pi), '--b', label = "Current Estimated Value")
    plt.legend(loc="upper right")
    plt.show()



