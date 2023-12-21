import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Advertising Prediction App")
st.write("This app predicts the **Advertising** sales!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.70, 296.4, 300.0)
    Radio = st.sidebar.slider('Radio', 0.0, 296.4, 30.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 200.0)

    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Sales Advertising.h5", "rb")) #rb: read binary
pred=loaded_model.predict(df)

st.subheader('Sales')
st.write(df)
