#importlas librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pickle


#importamos los datos en un formato dataframe de pandas
df_data=pd.read_csv("general_data.csv")

df_data_copy = df_data.copy()
df_data = df_data.dropna()

#codificación de variables categóricas
encoder=LabelEncoder()
col_names=df_data.columns

for col in col_names:
     if df_data[col].dtype=="object":
            df_data[col]=encoder.fit_transform(df_data[col])

#división de datos para análisis
x=df_data.drop('Attrition',axis=1)
y=df_data['Attrition']

#normalizamos con Standard Scaler
scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)

# Separamos los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# Inicialización del modelo de regresión logística
log_reg = LogisticRegression(C=1, max_iter=100)

# Entrenamiento del modelo de regresión logística
log_regr = log_reg.fit(x_train, y_train)

# Evaluación del rendimiento del modelo
st.markdown("#### Resultados del modelo de regresión logística")
st.write('--------------------------------------------------------------------------')
st.write('Logistic Regression:')
st.write('Training Model accuracy score: {:.3f}'.format(log_reg.score(x_train, y_train)))
st.write('Test Model accuracy score: {:.3f}'.format(log_reg.score(x_test, y_test)))
st.write('------------------------')


#Entrena el modelo KNeighbors cambiando el número de K vecinos para verificar el dato que genera los mejores resultados
best_k = 0
best_score = 0

for k in range(1, 10):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    
    train_accuracy = knn_classifier.score(x_train, y_train)
    test_accuracy = knn_classifier.score(x_test, y_test)
    
    st.write('-------------------------------------------------------------')
    st.write(f'Value of k is {k}')
    st.write('KNeighborsClassifier Classifier:')
    st.write('Training Model accuracy score: {:.3f}'.format(train_accuracy))
    st.write('Test Model accuracy score: {:.3f}'.format(test_accuracy))
    
    # Guardar el mejor modelo y k
    if test_accuracy > best_score:
        best_k = k
        best_score = test_accuracy

st.write('-------------------------------------------------------------')
st.write(f'Best k value: {best_k}')
st.write(f'Best test accuracy score: {best_score}')

# Entrenar el mejor modelo con el valor de k óptimo
best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_model = best_knn_classifier.fit(x_train, y_train)

#Guardo los datos de entrenamiento para la analítica posterior
with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open('knn_m.pkl', 'wb') as kn:
    pickle.dump(knn_model, kn)

