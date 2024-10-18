# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')


# Enter how many raindrops (random points) will be considered for the simulation
N = int(input('Enter the number of points/droplets for the interactions: '))

n_out = []  # Variable to count how many points were generated for the circle
n_in = []  # Variable to count how many points were generated inside the circle

# Counting the number of points inside the circle: 
for i in range(N):
    # Setting the input variables randomly in the range bounded by the square of side 2R
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    
    # If any point is generated inside the circle, record its coordinate (x, y) in the variable nc = []
    if  np.sqrt(x**2 + y**2) <= 1:
        n_in.append((x, y))
    else:
        n_out.append((x, y))
        
plt.figure(figsize = (6,6))

plt.scatter(
    [x[0] for x in n_in], 
    [x[1] for x in n_in],
    marker = ".", 
    alpha = 0.5)

plt.scatter(
    [x[0] for x in n_out],
    [x[1] for x in n_out],
    marker = ".",
    alpha = 0.5)

pi = 4 * float( len(n_in) / N)

print("\n Estimated value of pi: ", pi)

plt.show()