# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:55:00 2023

@author: sachi , majestichillary
"""

##correct method, which is a combination of method 01 and 02
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
 
# load the dataset
#def load_dataset():
#   data = pd.read_csv('imdb_edited.csv')
#    dataX=data[data.columns[10:15]]
#    dataY=data['Genre_cat']
#    X1=np.array(dataX)
    #keep the flollowing code segment under comment. we can use it later
    #data_normalized = preprocessing.normalize([X1])
    #row_norms=np.linalg.norm(X1, axis = 1, ord=1)
    #rn=row_norms.reshape(len(X1),1)
    #X=X1.reshape(len(X1),5)
    #data_normalized = X/rn
#    return X1, dataY

np.random.seed(1)
# load the dataset
data = pd.read_csv('imdb_edited.csv')
X=data[data.columns[10:15]]
y=data['Genre_cat']
#X, y = load_dataset()
#Split the data set into training data, test data and validation
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, stratify=y,test_size = 0.3,random_state=1)

# define the model
model = RandomForestClassifier(n_estimators=100,bootstrap=True,random_state=1)
#model = ExtraTreesClassifier(n_estimators=100,bootstrap=True,random_state=1)

classifier=model.fit(x_training_data,y_training_data)
predicted=classifier.predict(x_test_data)
#printing the results
print('Confusion Matrix :',confusion_matrix(y_test_data, predicted))
print ('Accuracy Score :',accuracy_score(y_test_data, predicted))
print ('Report : ',classification_report(y_test_data, predicted))

#ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100,bootstrap=True,random_state=1)
classifier=model.fit(x_training_data,y_training_data)
predicted=classifier.predict(x_test_data)
#printing the results
print('Confusion Matrix :',confusion_matrix(y_test_data, predicted))
print ('Accuracy Score :',accuracy_score(y_test_data, predicted))
print ('Report : ',classification_report(y_test_data, predicted))


#Support vector machine
model = SVC( kernel = 'linear' , C = 3 )
model.fit(x_training_data, y_training_data)
accuracy = model.score(x_test_data, y_test_data)
print (accuracy)


#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5 )
knn.fit(x_training_data, y_training_data)
knn_accuracy = knn.score(x_test_data, y_test_data)
print (knn_accuracy)
