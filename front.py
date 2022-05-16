import streamlit as st 
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

import pandas as pd

# set a title 
st.title("streamlit example") 


# add some text
st.write("""
# Explore diffrent classifier
""") 


# selection box input (title for the box , (op1 , ..., op2))
# dataset_name =st.selectbox("select Dataset" , ("iris","breast","wine")) 
# # RQ :  when we select an option from the box in the front ,the whole script  rerun again
# # RQ2 : si on veut faire un autre box , il faut que les options ne sont pas la meme pour le box précédent
# st.write(dataset_name)
# we can put the box as a side bar 
dataset_name= st.sidebar.selectbox("select Dataset" , ("IRIS","BREAST CANCER","WINE")) 


classifier_name= st.sidebar.selectbox("select Classifier" , ("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "IRIS" :
        data = datasets.load_iris()
    elif dataset_name =="BREAST CANCER" :
        data  = datasets.load_breast_cancer()
    else :
        data = datasets.load_wine()
    X= data.data
    y=data.target
    return X, y

X,y=get_dataset(dataset_name)

st.write("shape of dataset" , X.shape)
st.write("number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name) :
    params = dict()
    if clf_name == "KNN" :
        # sidebar : input ("name" , start  , end )
        K= st.sidebar.slider("K", 1 , 15 )
        params["K"] = K
    elif clf_name == 'SVM' :
        C= st.sidebar.slider("C", 0.01 , 10.0 )
        params["C"] = C
    else : 
        max_depth=st.sidebar.slider("max_depth", 2 , 15 )
        n_estimators=st.sidebar.slider("n_estimators", 1 , 100 )
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name , params) :
    if clf_name == "KNN" :
        clf= KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM' :
        clf=SVC(C=params['C'])
    else : 
        clf=RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=1234 )
    return clf

clf=get_classifier(classifier_name,params)

# classification : 

X_train , X_test , y_train ,y_test =train_test_split(X,y , test_size=0.2 , random_state=1234)

clf.fit(X_train , y_train)

y_pred=clf.predict(X_test )

acc= accuracy_score(y_test , y_pred)

st.write(f"classifier name = {classifier_name}")
st.write(f"acc = {acc}")


# PLOT 

pca = PCA(2)

X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
st.set_option('deprecation.showPyplotGlobalUse', False)
fig= plt.figure()
plt.scatter(x1 , x2 , c=y , alpha= 0.8 , cmap="viridis")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

st.pyplot()
# # Affiche data frame 
# df= pd.read_excel("module_crud/etudiant.xlsx")
# st.dataframe(df)

# text _ input 
nom= st.text_input(label='donner le nom')
st.write(nom)


# check box pour cliquer un box type retour boolean 

insert_one=st.checkbox("Insert one student") 