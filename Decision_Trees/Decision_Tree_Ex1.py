########################################################################################################
#
# You want to help your colleague decide whether or not to go to some comedy shows next month.
# Fortunately, he has made several notes in a notebook, noting certain characteristics about the comedian
# that helped him decide whether or not to go to previous shows.
# Based on the notes, create a decision tree that you can use to help your colleague decide
# whether or not to go to upcoming comedy shows.
########################################################################################################

# Importing the libraries
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import sys

root_path = Path(__file__).parent
csv_file_name = 'Comedy_Shows_Quality_Info.csv'
csv_file_path = root_path / 'csv_files' / csv_file_name 
# Reading the .csv file with the annotations and generating the dataframe
df = pd.read_csv(csv_file_path, sep = ';')

# Show the created dataframe
print(df)

# To build a Decision Tree, we need to transform all qualitative data into quantitative data, that is, we need to convert non-numeric data into numeric data. 

# We need to convert the data from the non-numeric columns "state_of_origin" and "have_already_gone_before?" into numeric data. 

# To do this conversion, we will use the map() method from the pandas library: 
# We will convert (replace):

# RJ -> 0, MG -> 1, SP -> 2

d_state = {'RJ': 0, 'MG': 1, 'SP': 2}
df['state_of_origin'] = df['state_of_origin'].map(d_state)

# Yes -> 1 e No -> 0
d_have_gone = {'Yes': 1, 'No': 0}
df['have_already_gone_before?'] = df['have_already_gone_before?'].map(d_have_gone)

# Shows what the dataframe looks like after transforming non-numeric data into numeric data
print('\n')
print(df)

# Next, we need to separate the columns of the predictor variables (with the characteristics) from the
# response column (target)

# We identify the predictor variables of the model
characteristics = ['comedian_age', 'carreer_experience_ages', 'show_quality_rank', 'state_of_origin']

# Agrupa as variáveis preditoras em X
X = df[characteristics] 

# Separate the response variable into y
y = df['have_already_gone_before?']

# Showing the separation of the predictor columns and the response column
print('\n')
print(X)
print(y)

# Builds the Decision Tree for the model
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names = characteristics)
plt.show()

###############################################################################################
#
#                   PREDICTING WHETHER OR NOT YOU WILL GO TO FUTURE SHOWS
#
###############################################################################################

while(True):
    print('\n')
    try:
        options = ['Q', 'C']

        option = input('Type Q to [Q]uit.\nType C to [C]ontinue.\n')
        option = option.capitalize()
        if option not in options:
            print('Please, type only one of the letters [Q] or [C].\n')
        elif option == 'Q':
            sys.exit()
        elif option == 'C':
            comedian_age = int(input('Enter the age of the comedian: '))
            if isinstance(comedian_age, int):
                
                carreer_experience_ages = int(input('Enter the carreer experience for the comedian in ages: '))
            
                if isinstance(carreer_experience_ages, int):
                    
                    show_quality_rank = int(input('Enter the show quality rank for the last perfomance: '))
            
                    if isinstance(show_quality_rank, int):
                        
                        state_of_origin = int(input('Enter the state of origin of the comedian: \nRJ -> 0\nMG -> 1\nSP -> 2\n\n'))
                        if isinstance(state_of_origin, int):
                            
                            should_I_go_to_the_show = dtree.predict(
                                        [
                                            [
                                                comedian_age, 
                                                carreer_experience_ages, 
                                                show_quality_rank, 
                                                state_of_origin
                                                ]
                                            ]
                                        )
                            print("\nShould I go to the show? ")
                            
                            if should_I_go_to_the_show == 1:
                                print('Answer: Yes')
                            else:
                                print('Answer: No')
    except Exception as e:
        print('You must type only integer numbers!. Try again.')
        print(e)

################################################################################################
#
#                               INTERPRETATION OF THE RESULT
#
################################################################################################
'''
The Decision Tree uses previous choices to calculate your colleague's next choices, whether or not he goes to the comedy show in this case. 

Each node or block of the Decision Tree presents the characteristics or predictor variables with their respective values: 

show_quality_rank <= 10.5 

This means that any comedian with points equal to or less than 10.5 will divert the analysis flow to the left arrow (True), and any point value above 10.5 will divert the analysis flow to the right (False). 

gini = 0.497 

The "gini" parameter is related to the quality of the analysis that resulted in the branching of a given node or block of the Decision Tree, always being a value between 0.0 and 0.5. 

gini = 0.0 --> Means that all analyses for the samples yield the same result (no branches are generated)

gini = 0.5 --> Means that half of the analyzed samples produce a True/Left branch
and the other half produce a False/Right branch

There are many ways to split the samples, we use the GINI method in this example.

The Gini method uses this formula:

Gini = 1 - (x/n)^2 - (y/n)^2

Where x is the number of positive responses ("Yes"),

n is the number of samples and

y is the number of negative responses ("No"),

which gives us this calculation:

1 - (7/13)^2 - (6/13)^2 = 0.497

samples = 13

Represents the number of samples in each node or block of the Decision Tree that will be analyzed to generate the subsequent branches

value = [6, 7]

Means that: Out of the 13 samples,

6 samples (shows) resulted in "No" (did not go to the show)

7 samples resulted in "Yes" (went to the show)

'''