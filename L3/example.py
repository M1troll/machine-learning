"""
1) Format data -> attribute selecting, data preparation
2) Fix missed values
    - Remove objects (if these objects not a lot)
    - avg value by column
    - random value from column
    - Find depend from another columns
    - (HARD) use valid rows for education and rows with misses for test
3) Removing wasteds
4) Сглаживание
5) Increase size


3 lab - k-nearest roommates

"""
# -*- coding: utf-8 -*-
"""
@author: beponikap
"""
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
import seaborn; seaborn.set() 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
#from sklearn.learning_curve import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import KFold

Allviborka = pd.read_csv('L3/iris.csv', sep=',')
XClass,YClass = np.hsplit(Allviborka, [4])
X_train, X_test, y_train, y_test = train_test_split(XClass, YClass, test_size=0.3, random_state=0)

N_train, _ = X_train.shape
N_test, _ = X_test.shape
print (N_train, N_test) 
n=10
param_grid = [{'metric':['chebyshev','manhattan','euclidean'], 'n_neighbors': range(1,n+1)}]
knn = KNeighborsClassifier()    
#GridSearchCV - МЕТОД ПОИСКА НАИЛУЧШЕГО НАБОРА ПАРАМЕТРОВ 
#доставляющих минимум ошибки по кросс валидации 
#cv-кол-во блоков 
grid = GridSearchCV(knn, param_grid, cv=10) 
#fit - настройка на данных (обучение)
grid.fit(X_train, y_train.iloc[:,0])
#score - оценка обученной модели
grid.score(X_test, y_test.iloc[:,0])

print("Наилучшие значения параметров: {}".format(grid.best_params_))
print("Наилучшее значение кросс-валидац. правильности: {:.2f}".format(grid.best_score_))
print("Наилучшая модель:\n{}".format(grid.best_estimator_))

result =  pd.DataFrame()
result['n_neighbors']=grid.cv_results_['param_n_neighbors']
result['metric']=grid.cv_results_['param_metric']
result['mean_test_score']=grid.cv_results_['mean_test_score']

plt.plot(result['n_neighbors'][0:n], result['mean_test_score'][0:n], 's', color = "red", label=result['metric'][0])
plt.plot(result['n_neighbors'][n:n*2], result['mean_test_score'][n:n*2], 'x', color = "black", label=result['metric'][n])
plt.plot(result['n_neighbors'][n*2:n*3], result['mean_test_score'][n*2:n*3], 'd', color = "blue", label=result['metric'][n*2])

plt.ylabel('Среднее значение по результатам кросс-валидации ')
plt.xlabel('n_neighbors')
plt.show()

best_ch = 0
best_man = 0
best_mink = 0
j=0

#j-номер строки
for j in range(10):

    if result['mean_test_score'][j] > best_ch:
        
        best_ch = result['mean_test_score'][j]
        k1=result['metric'][j]
        k2=result['n_neighbors'][j]

for j in range(10,20):
    if result['mean_test_score'][j] > best_man:
        best_man = result['mean_test_score'][j]
        k3=result['metric'][j]
        k4=result['n_neighbors'][j]
        
for j in range(20,30):
    if result['mean_test_score'][j] > best_mink:
        best_mink = result['mean_test_score'][j]
        k5=result['metric'][j]
        k6=result['n_neighbors'][j]    

print(best_ch)
print(best_man)
print(best_mink)
        
knn1 = KNeighborsClassifier(metric=k1,n_neighbors=k2)    
knn1.fit(X_train, y_train)
knn1.score(X_test, y_test)
print(knn1.score(X_test, y_test))

knn2 = KNeighborsClassifier(metric=k3,n_neighbors=k4)    
knn2.fit(X_train, y_train)
knn2.score(X_test, y_test)
print(knn2.score(X_test, y_test))

knn3 = KNeighborsClassifier(metric=k5,n_neighbors=k6)    
knn3.fit(X_train, y_train)
knn3.score(X_test, y_test)
print(knn3.score(X_test, y_test))

plt.plot(k2, best_ch, 's', color = "orange", label=k1)
plt.plot(k4, best_man, 's', color = "red", label=k3)
plt.plot(k6, best_mink, 's', color = "black", label=k5)

plt.ylabel('Точность')
plt.xlabel('n_neighbors')
plt.show()
