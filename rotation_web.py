#importlas librerías necesarias
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#extraemos los archivos pickle
with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)

with open('knn_m.pkl', 'rb') as kn:
    knn_m = pickle.load(kn)


#Función que transforma los datos para verificarlos en el modelo de machine learning
def transfor_data(df_data):
     encoder=LabelEncoder()
     col_names=df_data.columns
     for col in col_names:
          if df_data[col].dtype=="object":
               df_data[col]=encoder.fit_transform(df_data[col])

     #Dividimos los datos para el análisis
     x=df_data.drop('Attrition',axis=1)
     scaler=StandardScaler()
     x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
        
     return x


def rotation_by_department(x, df_human_records, df_data_copy, department, model_ml):
    
    # Utiliza el modelo entrenado para predecir las etiquetas del nuevo conjunto de datos
    if model_ml:
        predicted_labels = knn_m.predict(x)
    else:
        predicted_labels = log_reg.predict(x)

    # Agrega las etiquetas predichas al DataFrame original
    df_human_records['Predicted_Labels'] = predicted_labels
    
    # Suponiendo que 'df_human_records' es tu DataFrame
    attrition_count = (df_human_records['Predicted_Labels'] == 1).sum()

    if department == 1:
        st.markdown("#### Possible retirements in the human resources areas: --> " + str(attrition_count))
    elif department == 2:
        st.markdown("#### Possible retirements in the Research & Development area: --> " + str(attrition_count))
    else:
        st.markdown("#### Possible retirements in the sales area: --> " + str(attrition_count))
    
    # Filtrar los registros donde 'Attrition' es igual a 1
    df_attrition_1 = df_human_records[df_human_records['Predicted_Labels'] == 1]
    # Seleccionar solo los valores de la columna 'EmployeeID' y mantener el nombre de la columna
    df_employee_id = df_attrition_1['EmployeeID'].to_frame(name='EmployeeID')
    # Realizar el join
    result = pd.merge(df_employee_id, df_data_copy, on='EmployeeID', how='left')
    st.dataframe(result)

    return 


def main():
    #titulo
    st.title('Análisis de Rotación Temprana')
    st.sidebar.header('by @hector ocampo')
    #titulo de sidebar
    st.sidebar.header('User input data')
    
    #importamos los datos en un formato dataframe de pandas
    df_data=pd.read_csv("general_data.csv")
    #muestra el dataframe en la pantalla del browser
    st.dataframe(df_data)
    df_data_copy = df_data.copy()
    df_data = df_data.dropna()
    #reviso el modelo con datos reales
    df_data_copy2 = df_data_copy.copy()
    # cargo solo los registros con empleados que permanecen en la empresa
    df_data_copy2 = df_data_copy2[df_data_copy2['Attrition'] != 'Yes']
    
    #seleccionar el modelo de machine learning para el análisis
    option2 = ['Logistic Regression', 'KNeighbors Classifier']
    model2 = st.sidebar.selectbox('Select the machine learning model', option2)

    if model2 == 'Logistic Regression':
        st.write('Logistic Regression - Accuracy --> 0,85')
        model_ml = 0
    else:
        st.write('KNeighbors Classifier - Accuracy --> 0,98')
        model_ml = 1

    #seleccionar el área para análisis de rotación
    option = ['Human Resources', 'Research & Development', 'Saless']
    model = st.sidebar.selectbox('Select the area to analyze', option)
    
    if model == 'Human Resources':
        department = 1
        df_records = df_data_copy2[df_data_copy2['Department'] == 'Human Resources']
        df_human_records = df_records.dropna()
        x = transfor_data(df_human_records)
        rotation_by_department(x, df_human_records, df_data_copy, department, model_ml)
    elif model == 'Research & Development':
        department = 2
        df_records = df_data_copy2[df_data_copy2['Department'] == 'Research & Development']
        df_human_records = df_records.dropna()
        x = transfor_data(df_human_records)
        rotation_by_department(x, df_human_records, df_data_copy, department, model_ml)
    else:
        department = 3
        df_records = df_data_copy2[df_data_copy2['Department'] == 'Sales']
        df_human_records = df_records.dropna()
        x = transfor_data(df_human_records)
        rotation_by_department(x, df_human_records, df_data_copy, department, model_ml)
    
if __name__ == '__main__':
    main()    