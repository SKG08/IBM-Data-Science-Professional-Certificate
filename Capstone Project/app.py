# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 12:11:55 2021

@author: soumya
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, precision_score, recall_score
from sklearn.decomposition import PCA

st.set_option('deprecation.showPyplotGlobalUse', False)

# st.title("IBM Data Science Professional Certificate: Capstone Project")
st.write("""
         ### Performance Evaluation of Various Classification Algorithms
         """)
algo = st.sidebar.selectbox("Select Algorithm",("Logistic Regression","SVM",
                                 "Decision Tree","KNN"))

# Data Import

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')
# data.head()
X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')
# X.head(10)
y = data['Class'].to_numpy()
# y

transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

st.write("Shape of Input Data:", X.shape, "Shape of Target Data:", y.shape)


def get_params(algo):
    params = dict()
    if algo == "KNN":
        k = st.sidebar.slider("Neighbors(K):",1,20)
        w = st.sidebar.selectbox("Weights :",('uniform', 'distance'))
        m = st.sidebar.selectbox("Metric :",('euclidean', 'manhattan', 'minkowski'))
        params["k"] = k
        params["w"] = w
        params["m"] = m
    elif algo == "SVM":
        k = st.sidebar.selectbox("Kernel :",("linear","poly","rbf","sigmoid"))
        c = st.sidebar.slider("C :",0.01,10.0)
        params["c"] = c
        params["k"] = k
    elif algo == "Logistic Regression":
        s = st.sidebar.selectbox("Solver :",('newton-cg','lbfgs','liblinear'))
        if s == 'liblinear':
            p = st.sidebar.selectbox("Penalty :",('l1','l2'))
            params["p"] = p
        c = st.sidebar.select_slider("C :",(100.0, 10.0, 1.0, 0.1, 0.01))
        params["s"] = s
        params["c"] = c
    else:
        c = st.sidebar.selectbox("Criterion :",('gini','entropy'))
        d = st.sidebar.slider("Max Depth :", 1,10)
        ms = st.sidebar.slider("Min Sample Split :", 2,10)
        ml = st.sidebar.slider("Min Sample Leaf :",1,5)
        params["c"] = c
        params["d"] = d
        params["ms"] = ms
        params["ml"] = ml
    return params

params = get_params(algo)
    
# st.write(params["K"])

def get_model(algo,params):

    if algo == "KNN":
        model = KNeighborsClassifier(n_neighbors=params["k"],weights=params["w"],metric=params["m"])
    elif algo == "SVM":
        model = SVC(C=params["c"],kernel=params["k"])
    elif algo == "Logistic Regression":
        if params["s"] == 'liblinear':
            model = LogisticRegression(solver=params["s"], C=params["c"], penalty= params["p"])
        else:
            model = LogisticRegression(solver=params["s"], C=params["c"])
    else:
        model = DecisionTreeClassifier(criterion=params["c"],
                                       max_depth=params["d"],
                                       min_samples_split=params["ms"],
                                       min_samples_leaf =params["ml"])
    return model


model = get_model(algo, params)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=2)

# st.write('Train set shape:', X_train.shape,  y_train.shape)
# st.write('Test set shape:', X_test.shape,  y_test.shape)
# st.write("Train Success Rate:",np.count_nonzero(y_train==1)/y_train.shape[0]*100)
# st.write("Test Success Rate: ",np.count_nonzero(y_test==1)/y_test.shape[0]*100)

model.fit(X_train,y_train)

yhat = model.predict(X_test)

accuracy = accuracy_score(y_test,yhat)
precision = precision_score(y_test,yhat)
recall = recall_score(y_test,yhat)



st.write(f'Algorithm: {algo}')
st.write(f'Accuracy: {accuracy}')
st.write(f'Precision: {precision}')
st.write(f'Recall: {recall}')

# Visualization:



# Scatter Plot

pca = PCA(2)
x_pc = pca.fit_transform(X)

x1 = x_pc[:,0]
x2 = x_pc[:,1]


fig = plt.figure(figsize=(18, 7))
plt.scatter(x1, x2, c=y, alpha= 0.8, cmap="viridis")
plt.xlabel("Principal Component-1")
plt.ylabel("Principal Component-2")
plt.colorbar()

st.pyplot(fig)

# Confusion Matrix

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
    st.pyplot()
    
plot_confusion_matrix(y_test,yhat)
