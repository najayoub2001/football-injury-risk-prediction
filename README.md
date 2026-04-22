# football-injury-risk-prediction
Final Year Project – A machine learning system for predicting football injury risk using physiological and training data, featuring a Streamlit dashboard for risk analysis and player monitoring.

## Overview
This project develops and evaluates machine learning models to predict injury risk in football players based on physiological and workload data. 

A Random Forest model and an LSTM neural network were implemented and compared. The final system integrates the best-performing model into an interactive Streamlit dashboard.

## Features
- Injury risk prediction using machine learning
- Player dashboard with real-time updates
- Squad risk overview
- Scenario simulation using input sliders
- Player profile management with image upload and cropping

## Technologies Used
- Python
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Pandas / NumPy

## Dataset
University Football Injury Prediction Dataset (Kaggle):
https://www.kaggle.com/datasets/yuanchunhong/university-football-injury-prediction-dataset

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the app:
   streamlit run app.py

## Project Structure
- app.py → Main dashboard  
- utils/ → Helper functions  
- model/ → Trained model  
- assets/ → UI images  

## Author
Ayoub Naja
