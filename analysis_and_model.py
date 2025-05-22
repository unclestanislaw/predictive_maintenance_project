import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        label_encoder = LabelEncoder()
        data['Type'] = label_encoder.fit_transform(data['Type'])
        scaler = StandardScaler()
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]' ,'Torque [Nm]', 'Tool wear [min]']
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        # Визуализация результатов
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.text(classification_rep)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            productID = st.selectbox("productID", ["L", "M", "H"])
            air_temp = st.number_input("air temperature [K]")
            process_temp = st.number_input("process temperature [K]")
            rotational_speed = st.number_input("rotational speed [rpm]")
            torque = st.number_input("torque [Nm]")
            tool_wear = st.number_input("tool wear [min]")
            
            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                # Преобразование введенных данных
                input_data = pd.DataFrame({
                    'Type': [label_encoder.transform([productID])[0]],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotational_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                })
                
                # Масштабирование введенных данных
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])

                # Предсказание (дописать)
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]
                st.write(f"Предсказание: {prediction[0]}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")
    else:
        st.info("Пожалуйста, загрузите файл CSV для продолжения.")

if __name__ == "__main__":
    analysis_and_model_page()