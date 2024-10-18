############################################################################################
# What is the probability of flipping a coin 10 times and getting heads on 3 or more flips?
############################################################################################

# Importing the library used in analytical resolution
from scipy.stats import binom

# Importing the numpy library used in the Monte Carlo method
import numpy as np

###################################################################
#
# Analytical resolution by Binomial Distribution
#
###################################################################
plays = binom(n = 10, p = 0.5)

probability_calc = 1 - plays.cdf(3)

print("\n ******************************************************")
print("\n Value obtained analytically by Binomial Distribution")
print("\n ******************************************************")
print(f"\n Probability of tossing a coin 10 times and getting heads in more than 3 tosses: {probability_calc}")

##############################################################
#
# Value obtained by the Monte Carlo method
#
##############################################################
def perform_plays():
    
    # Generates 10 random values ​​between 0 and 1 (0% and 100%)
    plays = np.random.uniform(0, 1, 10)
    
    # Prints the result of each roll with 10 random values ​​between 0 and 1 (0% and 100%)
    # print("\n plays: ", plays)
    
    # print("\n Sum of plays with values greater than 0.5 (50 %):", (plays > 0.5).sum())
    
    # Return the sum of plays with values greater than 0.5 (50 %)
    return (plays > 0.5).sum()


print("\n ********************************************************")
print(f"\n Value obtained stochastically by the Monte Carlo method")
print("\n ********************************************************")
print(f"\n The probability of tossing a coin 10 times and getting heads on 3 or more tosses is equal to:")

# Number of plays to be performed.
N = [100, 1000, 10_000, 100_000]

for number_of_performances in N:
    count = 0
    for i in range(number_of_performances):
        # Counts the number of tosses whose result was greater than 3 "heads"
        if (perform_plays()) > 3:
            count += 1
        
    estimated_probability = float(count / number_of_performances)
    print(f"\n Prob = {estimated_probability}\t--> For N = {number_of_performances} performances")

print(
    '''\nAs we can see, as N increases, the probability obtained by the stochastic method approaches the probability obtained by the analytical method using a Binomial Distribution. Although the code for the stochastic method is relatively simple, we have to deal with the dilemma of more computational effort as N increases.
    ''')