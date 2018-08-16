import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(dataframe):

    dataframe.__delitem__('Alley')
    dataframe.Electrical = pd.get_dummies(dataframe['Electrical'])
    dataframe.MSZoning = pd.get_dummies(dataframe['MSZoning'])
    dataframe.LotShape = pd.get_dummies(dataframe['LotShape'])
    dataframe.LandContour = pd.get_dummies(dataframe['LandContour'])
    dataframe.Utilities = pd.get_dummies(dataframe['Utilities'])

    return dataframe

test_data = pd.read_csv('resources/train.csv')
pd.set_option('max_columns', 50)

test_data = preprocess_data(test_data)

print(test_data.describe())
print(test_data.info())
print(test_data.head())
print(test_data.corr()['SalePrice'].sort_values())
