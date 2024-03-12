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
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

np.random.seed(1)
# load the dataset
data = pd.read_csv('imdb_edited.csv')
X=data[data.columns[10:15]]
y=data['Genre_cat']
X1=np.array(X)
data_normalized = preprocessing.normalize(X, norm="l2")
from sklearn.feature_selection import SelectKBest, chi2
#data_normalized.shape
X_new = SelectKBest(chi2, k=5).fit_transform(X, y)
#X_new.shape

#Split the data set into training data, test data and validation
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X_new, y, stratify=y,test_size = 0.3,random_state=1)

# define the model
model = RandomForestClassifier(n_estimators=100,bootstrap=True,random_state=1)
classifier=model.fit(x_training_data,y_training_data)
predicted=classifier.predict(x_test_data)
#printing the results
print('Confusion Matrix :',confusion_matrix(y_test_data, predicted))
print ('Accuracy Score :',accuracy_score(y_test_data, predicted))
print ('Report : ',classification_report(y_test_data, predicted))
##########################################################################


#ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100,bootstrap=True,random_state=1)
classifier=model.fit(x_training_data,y_training_data)
predicted=classifier.predict(x_test_data)
#printing the results
print('Confusion Matrix :',confusion_matrix(y_test_data, predicted))
print ('Accuracy Score :',accuracy_score(y_test_data, predicted))
print ('Report : ',classification_report(y_test_data, predicted))

#Support vector machine
#model = SVC( kernel = 'linear' , C = 3 )
#model.fit(x_training_data, y_training_data)
#accuracy = model.score(x_test_data, y_test_data)
#print (accuracy)
##########################################################################


#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5 )
knn.fit(x_training_data, y_training_data)
knn_accuracy = knn.score(x_test_data, y_test_data)
print (knn_accuracy)
##########################################################################



##########################################################################
#Decision Tree
clf = DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=(1)) #max_depth is maximum number of levels in the tree
# Train Decision Tree Classifer
clf = clf.fit(x_training_data,y_training_data)
#Predict the response for test dataset
y_pred = clf.predict(x_test_data)
print ('Confusion Matrix :')
print(confusion_matrix(y_test_data, y_pred))
print ('Accuracy Score :',accuracy_score(y_test_data, y_pred))#accuracy is 88%

####check the optimum depth
max_depth_rng = list(range(1,12))
accuracy=[]
for depth in max_depth_rng:
    clf = DecisionTreeClassifier(criterion="gini",max_depth=(depth), random_state=(1))
    clf.fit(x_training_data,y_training_data)
    score = clf.score(x_test_data,y_test_data)
    accuracy.append(score)
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(max_depth_rng, accuracy)
ax.set(xlabel='maximum depth',ylabel='Accuracy')#,title='Optimizing the maximum depth')
ax.grid()
plt.show()
##########################################################################


##########################################################################
##Plots###

from sklearn.decomposition import NMF
import plotly.io as pio
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = "browser"

X_data = X / X.sum(axis=0)

model = NMF(n_components=3, init='random', solver='mu', max_iter=1000, random_state=1)
W = model.fit(X_data, y).transform(X_data)
ac = W[0:201,:]
cr = W[201:318,:]
dr = W[318:635,:]
oth = W[635:830,:]
fig=go.Figure()
fig.add_trace(go.Scatter3d(x=ac[:,0], y=ac[:,1], z=ac[:,2], name='Action',
                           mode='markers',marker_color='rgba(0, 0, 255, 1)' ))
fig.add_trace(go.Scatter3d(x=cr[:,0], y=cr[:,1], z=cr[:,2], name='Crime',
                           mode='markers',marker_color='rgba(255, 0, 0, 1)' ))

fig.add_trace(go.Scatter3d(x=dr[:,0], y=dr[:,1], z=dr[:,2], name='Drama',
                           mode='markers',marker_color='rgba(0, 255, 0, 1)' ))

fig.add_trace(go.Scatter3d(x=oth[:,0], y=oth[:,1], z=oth[:,2], name='Other',
                           mode='markers',marker_color='rgba(120, 0, 255, 0.5)' ))

fig.update_layout(title='3D Plot for IMDB Genres')
fig.show()
fig.write_html("3DNMF_IMDB.html")
##########################################################################


##########################################################################
#sort the dataset by date
#return with category based datasets
df_1 = data.loc[(data['Genre_cat'] == 1 )]
df_1= df_1.sort_values(by='Released_Year', ascending=True)
df_2 = data.loc[(data['Genre_cat'] == 2 )]
df_2= df_2.sort_values(by='Released_Year', ascending=True)
df_3 = data.loc[(data['Genre_cat'] == 3 )]
df_3= df_3.sort_values(by='Released_Year', ascending=True)
df_4 = data.loc[(data['Genre_cat'] == 4 )]
df_4= df_4.sort_values(by='Released_Year', ascending=True)


T1_1=df_1['Released_Year'].unique()
dts_1=[]
T2_1=df_1.groupby('Released_Year')['Runtime'].agg(np.mean)
tw_1=[]
T3_1=df_1.groupby('Released_Year')['IMDB_Rating'].agg(np.mean)
rtw_1=[]
T4_1=df_1.groupby('Released_Year')['Meta_score'].agg(np.mean)
lks_1=[]    
T5_1=df_1.groupby('Released_Year')['No_of_Votes'].agg(np.sum)
rpy_1=[] 
T6_1=df_1.groupby('Released_Year')['Gross'].agg(np.sum)
gr_1=[] 
T7_1=df_1.groupby('Released_Year')['Genre_cat'].agg(np.max)
cat_1=[]
s=0
for i in T1_1:
    dts_1.append(i)
    tw_1.append(T2_1[s])
    rtw_1.append(T3_1[s])
    lks_1.append(T4_1[s])
    rpy_1.append(T5_1[s]) 
    gr_1.append(T6_1[s])
    cat_1.append(T7_1[s])
    s=s+1
    
T1_2=df_2['Released_Year'].unique()
dts_2=[]
T2_2=df_2.groupby('Released_Year')['Runtime'].agg(np.mean)
tw_2=[]
T3_2=df_2.groupby('Released_Year')['IMDB_Rating'].agg(np.mean)
rtw_2=[]
T4_2=df_2.groupby('Released_Year')['Meta_score'].agg(np.mean)
lks_2=[]    
T5_2=df_2.groupby('Released_Year')['No_of_Votes'].agg(np.sum)
rpy_2=[] 
T6_2=df_2.groupby('Released_Year')['Gross'].agg(np.sum)
gr_2=[] 
T7_2=df_2.groupby('Released_Year')['Genre_cat'].agg(np.max)
cat_2=[]   
s=0
for i in T1_2:
    dts_2.append(i)
    tw_2.append(T2_2[s])
    rtw_2.append(T3_2[s])
    lks_2.append(T4_2[s])
    rpy_2.append(T5_2[s]) 
    gr_2.append(T6_2[s])
    cat_2.append(T7_2[s])
    s=s+1    
    
T1_3=df_3['Released_Year'].unique()
dts_3=[]
T2_3=df_3.groupby('Released_Year')['Runtime'].agg(np.mean)
tw_3=[]
T3_3=df_3.groupby('Released_Year')['IMDB_Rating'].agg(np.mean)
rtw_3=[]
T4_3=df_3.groupby('Released_Year')['Meta_score'].agg(np.mean)
lks_3=[]    
T5_3=df_3.groupby('Released_Year')['No_of_Votes'].agg(np.sum)
rpy_3=[] 
T6_3=df_3.groupby('Released_Year')['Gross'].agg(np.sum)
gr_3=[] 
T7_3=df_3.groupby('Released_Year')['Genre_cat'].agg(np.max)
cat_3=[]   
s=0
for i in T1_3:
    dts_3.append(i)
    tw_3.append(T2_3[s])
    rtw_3.append(T3_3[s])
    lks_3.append(T4_3[s])
    rpy_3.append(T5_3[s]) 
    gr_3.append(T6_3[s])
    cat_3.append(T7_3[s])
    s=s+1  

T1_4=df_4['Released_Year'].unique()
dts_4=[]
T2_4=df_4.groupby('Released_Year')['Runtime'].agg(np.mean)
tw_4=[]
T3_4=df_4.groupby('Released_Year')['IMDB_Rating'].agg(np.mean)
rtw_4=[]
T4_4=df_4.groupby('Released_Year')['Meta_score'].agg(np.mean)
lks_4=[]    
T5_4=df_4.groupby('Released_Year')['No_of_Votes'].agg(np.sum)
rpy_4=[] 
T6_4=df_4.groupby('Released_Year')['Gross'].agg(np.sum)
gr_4=[] 
T7_4=df_4.groupby('Released_Year')['Genre_cat'].agg(np.max)
cat_4=[]   
s=0
for i in T1_4:
    dts_4.append(i)
    tw_4.append(T2_4[s])
    rtw_4.append(T3_4[s])
    lks_4.append(T4_4[s])
    rpy_4.append(T5_4[s]) 
    gr_4.append(T6_4[s])
    cat_4.append(T7_4[s])
    s=s+1  
    
from sklearn.decomposition import NMF
import plotly.io as pio
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = "browser"    
fig = go.Figure()
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig.add_trace(go.Bar(x=dts_1, y=tw_1,name='Runtime',legendgroup="Runtime",
                     marker_color='blue',opacity=1),
              row=1, col=1)
fig.add_trace(go.Bar(x=dts_1, y=rtw_1,name='IMDB_Rating',legendgroup="IMDB_Rating",
                     marker_color='red',opacity=1),
              row=1, col=1)
fig.add_trace(go.Bar(x=dts_1, y=lks_1,name='Meta_score',legendgroup="Meta_score",
                     marker_color='green',opacity=1),
              row=1, col=1)
fig.add_trace(go.Bar(x=dts_1, y=rpy_1,name='No_of_Votes',legendgroup="No_of_Votes",
                     marker_color='purple',opacity=1),
              row=1, col=1)
fig.add_trace(go.Bar(x=dts_1, y=gr_1,name='Gross',legendgroup="Gross",
                     marker_color='purple',opacity=1),
              row=1, col=1)     


fig.add_trace(go.Bar(x=dts_2, y=tw_2,name='Runtime',legendgroup="Runtime",
                     marker_color='blue',opacity=1),
              row=2, col=1)
fig.add_trace(go.Bar(x=dts_2, y=rtw_2,name='IMDB_Rating',legendgroup="IMDB_Rating",
                     marker_color='red',opacity=1),
              row=2, col=1)
fig.add_trace(go.Bar(x=dts_2, y=lks_2,name='Meta_score',legendgroup="Meta_score",
                     marker_color='green',opacity=1),
              row=2, col=1)
fig.add_trace(go.Bar(x=dts_2, y=rpy_2,name='No_of_Votes',legendgroup="No_of_Votes",
                     marker_color='purple',opacity=1),
              row=2, col=1)
fig.add_trace(go.Bar(x=dts_2, y=gr_2,name='Gross',legendgroup="Gross",
                     marker_color='purple',opacity=1),
              row=2, col=1)       

fig.add_trace(go.Bar(x=dts_3, y=tw_3,name='Runtime',legendgroup="Runtime",
                     marker_color='blue',opacity=1),
              row=3, col=1)
fig.add_trace(go.Bar(x=dts_3, y=rtw_3,name='IMDB_Rating',legendgroup="IMDB_Rating",
                     marker_color='red',opacity=1),
              row=3, col=1)
fig.add_trace(go.Bar(x=dts_3, y=lks_3,name='Meta_score',legendgroup="Meta_score",
                     marker_color='green',opacity=1),
              row=3, col=1)
fig.add_trace(go.Bar(x=dts_3, y=rpy_3,name='No_of_Votes',legendgroup="No_of_Votes",
                     marker_color='purple',opacity=1),
              row=3, col=1)
fig.add_trace(go.Bar(x=dts_3, y=gr_3,name='Gross',legendgroup="Gross",
                     marker_color='purple',opacity=1),
              row=3, col=1)

fig.add_trace(go.Bar(x=dts_4, y=tw_4,name='Runtime',legendgroup="Runtime",
                     marker_color='blue',opacity=1),
              row=4, col=1)
fig.add_trace(go.Bar(x=dts_4, y=rtw_4,name='IMDB_Rating',legendgroup="IMDB_Rating",
                     marker_color='red',opacity=1),
              row=4, col=1)
fig.add_trace(go.Bar(x=dts_4, y=lks_4,name='Meta_score',legendgroup="Meta_score",
                     marker_color='green',opacity=1),
              row=4, col=1)
fig.add_trace(go.Bar(x=dts_4, y=rpy_4,name='No_of_Votes',legendgroup="No_of_Votes",
                     marker_color='purple',opacity=1),
              row=4, col=1)
fig.add_trace(go.Bar(x=dts_4, y=gr_4,name='Gross',legendgroup="Gross",
                     marker_color='purple',opacity=1),
              row=4, col=1) 

# Update yaxis properties
fig.update_yaxes(title_text="Action", row=1, col=1)
fig.update_yaxes(title_text="Crime",  row=2, col=1)
fig.update_yaxes(title_text="Drama",  row=3, col=1)
fig.update_yaxes(title_text="Other", row=4, col=1)

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                  title_text="IMDB movies summary during the time")
fig.show()
fig.write_html("IMDB_activity_summary.html")
##########################################################################
