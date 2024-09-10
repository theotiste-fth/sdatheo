import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer, precision_score, precision_recall_curve
from sklearn.metrics import  roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


st.set_option("deprecation.showPyplotGlobalUse", False)
def main():
  st.title(" Application de machine learning pour le risque de crédit")
  st.subheader("Auteur : Franck Théotiste")
  
# Fonction d'importation des données

def load_data():
  data = pd.read_csv('Loan_data.csv')  
  return data
# Affichage de la table des données
df = load_data()
df_sample = df.sample(100)
if st.sidebar.checkbox("Afficher les données brutes", False):
   st.subheader("Jeu de données 'Risque de crédit' : Echantillon  de 100 observations")
   st.write(df_sample)

y = df['default']
X = df.drop('default', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 42)
 #  X_train, X_test, y_train, y_test = split(df)
                      
classifier = st.sidebar.selectbox("classificateur", ("Random Forest",  "Logistic Regression"))
  
#Analyse de la performance des modèles 
def plot_perf(graphes):
     if 'confusion matrix' in graphes:
       st.subheader('Matrice de confusion')
       ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
     st.pyplot()
  
     if 'ROC curve' in graphes:
       st.subheader('Courbe ROC')
       RocCurveDisplay.from_estimator(model, X_test, y_test)
     st.pyplot()
  
     if 'Precision-Recall curve' in graphes:
       st.subheader('Courbe Precision-Recall')
       PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
       st.pyplot()
      
#Random Forest
if classifier == "Random Forest":
   st.sidebar.subheader("Hyperparamètres du modèle")
   n_arbres = st.sidebar.number_input("Choisir le nombre d'arbres dans la forêt", min_value=100, max_value=1000, format="%g",step=10, key='n_estimators')   
   profondeur_arbre = st.sidebar.number_input("Choisir la profondeur maximale d'un arbre", min_value=1, max_value=20,format="%g", step=1)
   bootstrap=st.sidebar.radio("Echantillon bootstrap lors de la création d'arbres ?", ("True","False")) 
   
   Graph_perf = st.sidebar.multiselect("Choisir un graphique de performance du modèle ML", ("confusion matrix", "ROC curve", "Precision-Recall curve")) 
   if st.sidebar.button("Exécution", key="classify"):
      st.subheader("Random Forest Results")
      #Initialisation d'un objet RandomForestClassifier
      bootstrap = True if bootstrap == "True" else "False"
      if bootstrap == "True":
         bootstrap = True
      else:
          bootstrap = False    
      model=RandomForestClassifier(n_estimators=n_arbres, max_depth=profondeur_arbre, bootstrap=bootstrap, random_state=42)


      #Entrainement de l'algorithme
      model.fit(X_train, y_train)
      #Prédictions
      y_pred = model.predict(X_test)
      #Métriques de performances
      accuracy = model.score(X_test, y_test)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      #Afficher les métriques dans l'application
      st.write("Accuracy :", round(accuracy, 3))
      st.write("Precision :", round(precision, 3))
      st.write("Recall :", round(recall, 3))
      #st.write("Recall :", recall.round(3))
      #Afficher les graphiques de performances
      plot_perf(Graph_perf)



#Régression logistique
if classifier == "Logistic Regression":
   st.sidebar.subheader("Hyperparamètres du modèle")
   hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", min_value=1, max_value=10, format="%g")   
   n_max_iter = st.sidebar.number_input("Choisir le nombre maximun d'itérations", min_value=100, max_value=1000, format="%g", step=10)
     
   Graph_perf = st.sidebar.multiselect("Choisir un graphique de performance du modèle ML", ("confusion matrix", "ROC curve", "Precision-Recall curve")) 
   if st.sidebar.button("Exécution", key="classify"):
      st.subheader("Logistic Regression Results")
      #Initialisation d'un objet LogisticRegression
      model=LogisticRegression(C=hyp_c, max_iter=n_max_iter, random_state=42)
      #Entrainement de l'algorithme
      model.fit(X_train, y_train)
      #Prédictions
      y_pred = model.predict(X_test)
      #Métriques de performances
      accuracy = model.score(X_test, y_test)
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      #Afficher les métriques dans l'application
      st.write("Accuracy :", round(accuracy, 3))
      st.write("Precision :", round(precision, 3))
      st.write("Recall :", round(recall, 3))
      
      #Afficher les graphiques de performances
      plot_perf(Graph_perf)

if __name__ == '__main__':
   main()  