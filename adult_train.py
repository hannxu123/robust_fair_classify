
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

## preprocess the dataset
dataset = pd.read_csv('./dataset/adult.csv')
dataset = dataset.replace('?', np.nan)
dataset.dropna(inplace = True)

## change label
for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])
X = dataset.drop('income', axis=1)
Y = dataset['income']

print(dataset.head())