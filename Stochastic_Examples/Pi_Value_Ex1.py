from pickletools import int4
import numpy as np
import random
import matplotlib.pyplot as plt


# Define the lines of the square:
horiz = np.array(range(100))/100.0
y_1 = np.ones(100)
plt.plot(horiz , y_1, 'b')
vert = np.array(range(100))/100.0
x_1 = np.ones(100)
plt.plot(x_1 , vert, 'b')

# Resets ("clears") the droplet count in the circle area before starting the simulation
n_in = 0

# Control variable of the while() loop, starts at i = 1 and is incremented until it reaches the total number of drops nc 
i = 1

# Enter how many raindrops (random points) will be considered for the simulation
N = int(input('Enter the number of points/droplets for the interactions: '))

# Distribute the raindrops (random points) over the area of ​​the square:
while ( i <= N ):
  
  x = random.random()
  y = random.random()
  
  # If a drop fell inside the circle, then count that drop and show its coordinate.
  if np.sqrt((x**2) + (y**2)) <= 1:
    n_in += 1
    plt.plot(x , y , 'bo')
    
  # If a drop fell outside the circle, it only shows where it fell.
  else:
    plt.plot(x , y , 'ro')
  i += 1

# Calculates the estimated pi value by the ratio between the number of drops inside the circle and the total number of drops
pi = float ((4 * n_in) / N)

print ("O valor de pi é: ", pi)

plt.show()