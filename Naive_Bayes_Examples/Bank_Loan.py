########################################################################################################
#   
# A certain bank has the loan history data, as seen in the table below. With this data,
# the bank requested that a model be built that, by providing the input data, indicates whether or not to
# provide the loan
#
#########################################################################################################

# Importing the libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

# Generating the dataset
data = {'Income': ['High', 'Intermediary', 'Intermediary', 'Low', 'Low', 'Low', 'Low', 'High', 'Low', 'Intermediary'], 
         'Age': ['Young', 'Elderly', 'Adult', 'Adult', 'Adult', 'Elderly', 'Young', 'Young', 'Young', 'Young'],
         'Loan_Value': ['High', 'High', 'Intermediary','Intermediary', 'Intermediary', 'Low', 'High', 'Intermediary', 'Low', 'Low'],
         'Took_out_a_loan': ['Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes','Yes']} 

data = pd.DataFrame(data = data)

# Creating the LabelEncoder
Income_lbencoder = preprocessing.LabelEncoder()
Age_lbencoder = preprocessing.LabelEncoder()
value_lbencoder = preprocessing.LabelEncoder()
took_out_a_loan = preprocessing.LabelEncoder()

# Using LabelEncoder to assign numbers to qualitative variables
Income_lbencoder.fit(data['Income'].unique())
Age_lbencoder.fit(data['Age'].unique())
value_lbencoder.fit(data['Loan_Value'].unique())
took_out_a_loan.fit(data['Took_out_a_loan'].unique())

# Transforming the dataset from qualitative variables to quantitative variables
data['Income'] = Income_lbencoder.transform(data['Income'])
data['Age'] = Age_lbencoder.transform(data['Age'])
data['Loan_Value'] = value_lbencoder.transform(data['Loan_Value'])
data['Took_out_a_loan'] = took_out_a_loan.transform(data['Took_out_a_loan'])

# Separating our dataset into predictor attributes and the objective class
forecaster = data[['Income','Age','Loan_Value']]
classifier = data['Took_out_a_loan']

# Creating the Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(forecaster, classifier)

# Verificando a precisão
print("\n Precisão = ", gnb.score(forecaster, classifier)*100,"%")

# Inserting new dates to be predicted
forecast = {'Income': ['Intermediary', 'High'], 'Age': ['Young', 'Young'], 'Loan_Value':['Low', 'High']}
forecast = pd.DataFrame(data = forecast)

forecast['Income'] = Income_lbencoder.transform(forecast['Income'])
forecast['Age'] = Age_lbencoder.transform(forecast['Age'])
forecast['Loan_Value'] = value_lbencoder.transform(forecast['Loan_Value'])

# Checking the result
print("\n", gnb.predict(forecast))
print("\n", took_out_a_loan.inverse_transform(gnb.predict(forecast)))

# Verificando as probabilAges
print("\n", gnb.predict_proba(forecast))




