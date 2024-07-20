import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(current_dir, 'model.pkl')

# Load the model with error handling
try:
    model = pk.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error(f"Error: 'model.pkl' file not found at {model_path}")
    st.stop()  # Stop execution if the model file is not found
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()  # Stop execution if an unexpected error occurs

# Header for the Streamlit app
st.header('Car Price Prediction ML Model')

# Load and process car details
try:
    cars_data = pd.read_csv(os.path.join(current_dir, 'Cardetails.csv'))
except FileNotFoundError:
    st.error(f"Error: 'Cardetails.csv' file not found at {os.path.join(current_dir, 'Cardetails.csv')}")
    st.stop()  # Stop execution if the file is not found
except Exception as e:
    st.error(f"An unexpected error occurred while loading the CSV file: {e}")
    st.stop()  # Stop execution if an unexpected error occurs

# Preprocess car data
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit widgets
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Seller type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

# Predict button
if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )
    
    # Encode categorical variables
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                                       [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], inplace=True)

    try:
        car_price = model.predict(input_data_model)
        st.markdown(f'Car Price is going to be {car_price[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
