############################################################################################
#
#  Calculating the average age of The Simpsons characters
# 
###########################################################################################

import pandas as pd
import numpy as np

group_1 = (1, 8, 10, 38, 39)

group_2 = (8, 10, 39, 45, 49)

avg_age_group_1 = np.mean(group_1)

avg_age_group_2 = np.mean(group_2)


print ("\n Average for ages of group 1 = ", avg_age_group_1)

print ("\n Average for ages of group 2 = ", avg_age_group_2)

