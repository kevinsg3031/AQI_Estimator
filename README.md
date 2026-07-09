# AQI Lens 

Live Demo: https://aqi-estimator-eight.vercel.app/

AQI Lens is an interactive air quality analytics platform that combines machine learning, explainable AI, and real-time environmental monitoring to help users understand and explore air quality trends across Indian cities.

## Features

###  AQI Prediction
Predict Air Quality Index (AQI) values for major Indian cities using a trained machine learning model built on historical environmental data.

###  City Comparison Dashboard
Compare AQI trends across multiple cities through interactive visualizations and historical trend analysis.

###  Explainable AI
Understand model behavior through feature importance analysis and explainability visualizations, helping users interpret the factors influencing AQI predictions.

###  Live AQI Leaderboards
View real-time rankings of:
- Cleanest major cities
- Most polluted major cities

Live AQI data is fetched dynamically through external air-quality APIs.

### 📈 Model Performance Insights
Explore key machine learning evaluation metrics including:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

###  Modern User Interface
- Responsive design
- Dark/Light mode
- Interactive charts powered by Chart.js
- Single-page dashboard experience

---

## Project Architecture

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Explainability Analysis
      ↓
JSON Export
      ↓
Interactive Web Dashboard
