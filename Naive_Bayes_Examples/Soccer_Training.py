########################################################################################################
#   
# A football club has historical data on weather conditions for some previous training sessions. 
# In an attempt to save time and money, the club president decides to hire a consultancy to assess whether 
# the team will be able to train in the upcoming weekends based on weather forecasts obtained from the main 
# regional news outlets. 
# 
# The data scientist from the consulting firm hired decides to start the analysis by building a predictive 
# model using the Naive Bayes method.
#
#########################################################################################################

# Importing the libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

# Gerando o dataset
dados = {'Weather_Conditions': ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Rainy', 
                   'Cloudy', 'Sunny', 'Sunny', 'Rainy'], 
         'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 
                     'Normal', 'Normal'],
         'Wind_Velocity': ['Weak', 'Strong', 'Weak','Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
         'Team_Has_Triained': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes','Yes']} 

dados = pd.DataFrame(data = dados)

# Creating the LabelEncoder
Weather_Conditions_lbencoder = preprocessing.LabelEncoder()
Humidity_lbencoder = preprocessing.LabelEncoder()
Wind_Velocity_lbencoder = preprocessing.LabelEncoder()
Team_Has_Triained_lbencoder = preprocessing.LabelEncoder()

# Using LabelEncoder to assign numbers to qualitative variables
Weather_Conditions_lbencoder.fit(dados['Weather_Conditions'].unique())
Humidity_lbencoder.fit(dados['Humidity'].unique())
Wind_Velocity_lbencoder.fit(dados['Wind_Velocity'].unique())
Team_Has_Triained_lbencoder.fit(dados['Team_Has_Triained'].unique())

# Transforming the dataset from qualitative variables to quantitative variables
dados['Weather_Conditions'] = Weather_Conditions_lbencoder.transform(dados['Weather_Conditions'])
dados['Humidity'] = Humidity_lbencoder.transform(dados['Humidity'])
dados['Wind_Velocity'] = Wind_Velocity_lbencoder.transform(dados['Wind_Velocity'])
dados['Team_Has_Triained'] = Team_Has_Triained_lbencoder.transform(dados['Team_Has_Triained'])

# Separating our dataset into predictor attributes and the objective class
previsor = dados[['Weather_Conditions','Humidity','Wind_Velocity']]
classe = dados['Team_Has_Triained']

# Creating the NaiveBayes classifier
gnb = GaussianNB()
gnb.fit(previsor, classe)

# Checking accuracy
print("\n Precision = ", gnb.score(previsor, classe)*100,"%")

# Inserting new data to be predicted
forecast = {'Weather_Conditions': ['Sunny', 'Cloudy', 'Cloudy','Rainy'], 
            'Humidity': ['Normal', 'High','Normal','High'], 
            'Wind_Velocity':['Strong', 'Strong', 'Weak', 'Strong']}

forecast = pd.DataFrame(data = forecast)

forecast['Weather_Conditions'] = Weather_Conditions_lbencoder.transform(forecast['Weather_Conditions'])
forecast['Humidity'] = Humidity_lbencoder.transform(forecast['Humidity'])
forecast['Wind_Velocity'] = Wind_Velocity_lbencoder.transform(forecast['Wind_Velocity'])

print('\nChecking the result')
print("\n", gnb.predict(forecast))
print("\n", Team_Has_Triained_lbencoder.inverse_transform(gnb.predict(forecast)))

print('\nChecking the odds')
print("\n", gnb.predict_proba(forecast))




