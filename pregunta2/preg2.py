# -*- coding: utf-8 -*-
"""
@author: Alvaro
"""
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
cpu = pd.read_csv('cpu.csv')
print(cpu.head())
print(cpu.describe().transpose())
print(cpu.shape)
#target_column = ['class'] 
#norm
#predictors = list(set(list(cpu.columns))-set(target_column))
#cpu[predictors] = cpu[predictors]/cpu[predictors].max()
#print(cpu.describe().transpose())
X = cpu.iloc[:,:-1].values
y = cpu['class'].values
print(y.shape)
print(X.shape)


model = MLPClassifier(hidden_layer_sizes=(6,12,24) , max_iter=10000)
accuracy_list=[]
for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
  model.fit(X_train, y_train)
  y_esp=model.predict(X_test)
  #print(y_esp)
  #print(y_test)
  cm = confusion_matrix(y_test, y_esp)
  print(cm)
  accuracy=accuracy_score(y_test, y_esp)
  print(accuracy)
  accuracy_list.append(accuracy)
print(accuracy_list)
print("----------")
print(np.median(accuracy_list))