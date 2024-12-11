import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('train_filtered.csv')

X = data.drop(columns=['target']) # Признаки
Y = data['target'] # То, что нам нужно

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = lightgbm.LGBMRegressor(random_state=42)
model.fit(X_train, Y_train)