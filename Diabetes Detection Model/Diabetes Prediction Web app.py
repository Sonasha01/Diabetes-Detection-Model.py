# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:34:17 2024

@author: SOSA
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/SOSA/OneDrive/clgggg/Desktop/Diabetes Detection Model/trained_model.sav', 'rb'))


#creating a function for prediction

def diabetes_prediction(input_data):
    
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'


def main():
    
    
    #giving a title
    st.title ('Diabetes Prediction Web Page')

#getting title the input from the user
	
Pregnancies = st.text_input('Number of Pregancies')
Glucose = st.text_input('Glucose Level')
 
BloodPressure = st.text_input('Blood Pressure Value')
SkinThickness = st.text_input('Skin Thickness Value')
Insulin = st.text_input('Insulin Level')
BMI = st.text_input('BMI Level')
DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
Age = st.text_input('Age of the person')



#code for prediction
diagnosis = ' '
# creating a button for Prediction 
if st.button('Diabetes Test Result'):
 diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
st.success(diagnosis)

if  __name__ == '__main__':
    main()
    
 





















