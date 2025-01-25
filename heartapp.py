import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Train Model
@st.cache_resource
def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, scaler

# Prediction
def predict(model, scaler, user_input):
    user_input = scaler.transform([user_input])
    prediction = model.predict(user_input)
    return prediction

# App Title
st.title('Heart Disease Prediction App')

# Upload CSV Data
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write('Data Preview:')
    st.dataframe(data.head())

    # Train Model Button
    if st.button('Train Model'):
        model, X_test, y_test, scaler = train_model(data)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['columns'] = data.columns[:-1]
        st.success('Model Trained Successfully!')
        
        # Display Feature Importance
        st.subheader('Feature Importance')
        feature_importance = pd.Series(model.feature_importances_, index=data.columns[:-1])
        st.bar_chart(feature_importance.sort_values(ascending=False))
        
        # Display Model Accuracy
        accuracy = model.score(X_test, y_test)
        st.write(f'Model Accuracy: {accuracy:.2f}')

# Prediction Form
if 'model' in st.session_state:
    st.subheader('Make a Prediction')
    user_input = []

    if 'user_inputs' not in st.session_state:
        st.session_state['user_inputs'] = [0] * len(st.session_state['columns'])

    for i, col in enumerate(st.session_state['columns']):
        user_input.append(st.number_input(f'Input {col}', value=st.session_state['user_inputs'][i], key=col))

    if st.button('Predict'):
        st.session_state['user_inputs'] = user_input
        prediction = predict(st.session_state['model'], st.session_state['scaler'], user_input)
        st.write(f'Prediction : {"Heart Disease !! Please Consult a Doctor" if prediction[0] == 1 else "No Heart Disease"}')
