# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:35:51 2022

@author: Acer
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv("winequality-white.csv", sep=";")
print(df.head())

df.dropna()

X = df[["alcohol", "volatile acidity", "sulphates", "total sulfur dioxide"]]
y = df[["quality"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

lr_model = LinearRegression().fit(X_train, y_train)

print(lr_model.score(X_test, y_test))



joblib.dump(lr_model, "linear_regression_model.pkl")