# AQI Lens 
### Explainable Machine Learning for Air Quality Prediction and Analysis

🔗 **Live Demo:** https://aqi-estimator-eight.vercel.app/

AQI Lens is an interactive machine learning-powered dashboard that predicts and analyzes Air Quality Index (AQI) across major Indian cities. The platform combines predictive analytics, explainable AI, real-time AQI monitoring, and comparative visualizations to help users better understand environmental conditions and the factors influencing air quality.

---

## Overview

Air pollution remains one of the most significant environmental challenges in urban India. AQI Lens was developed to explore how machine learning can be used to model and predict air quality trends while maintaining transparency through explainability techniques.

The project provides:

- AQI prediction using a trained machine learning model
- Interactive city-wise AQI comparison
- Explainable AI visualizations
- Live AQI leaderboards
- Historical trend exploration
- Responsive and modern web interface

---

## Key Features

###  AQI Prediction

Generate AQI predictions for major Indian cities using a trained K-Nearest Neighbors regression model.

The prediction interface provides:

- Predicted AQI values
- AQI category classification
- City-specific air quality insights
- Interactive visualization of results

---

###  City Comparison Dashboard

Compare air quality patterns across multiple cities using interactive charts.

Features include:

- Side-by-side AQI trend comparison
- Historical monthly AQI visualization
- Comparative environmental analysis
- Interactive Chart.js visualizations

---

###  Explainable AI

AQI Lens emphasizes interpretability alongside prediction accuracy.

The Explainability module provides:

- Feature importance analysis
- Impact visualization of environmental factors
- Model transparency insights
- Understanding of prediction drivers

This enables users to understand *why* a prediction was generated rather than simply viewing the predicted value.

---

###  Live AQI Leaderboards

Real-time AQI data is fetched dynamically to display:

- Cleanest major Indian cities
- Most polluted major Indian cities

The leaderboard updates using live environmental data sources, providing a snapshot of current air quality conditions.

---

###  Modern User Experience

- Dark / Light mode
- Fully responsive design
- Interactive dashboards
- Real-time data visualization
- Single-page application architecture

---

# Machine Learning Pipeline

## Dataset Preparation

The dataset was preprocessed to ensure consistency and reliability before training.

Key preprocessing steps included:

- Data cleaning
- Handling missing values
- Feature selection
- Dataset transformation
- Creation of model-ready feature vectors

---

## Features Used

The model learns AQI patterns using environmental and city-level information, including:

- PM2.5 concentration
- PM10 concentration
- Nitrogen Dioxide (NO₂)
- Sulfur Dioxide (SO₂)
- Carbon Monoxide (CO)
- Ozone (O₃)
- Temperature
- Humidity
- Seasonal information
- City-specific characteristics

These features collectively capture the environmental conditions that influence air quality.

---

## Model Selection

Multiple machine learning algorithms were evaluated during experimentation.

Models explored included:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor

After evaluation, the **K-Nearest Neighbors (KNN) Regressor** produced the strongest overall performance and was selected as the final model.

### Why KNN?

KNN predicts AQI by identifying historical observations that are most similar to the current environmental conditions.

For a new prediction:

1. The model calculates the distance between the input and historical observations.
2. The nearest neighbors are identified.
3. The AQI values of those neighbors are aggregated.
4. The average of the neighboring AQI values becomes the prediction.

This approach allows the model to capture local patterns and relationships in environmental data without assuming a fixed mathematical relationship between variables.

---

## Model Evaluation

The model was evaluated using standard regression metrics:

### Mean Absolute Error (MAE)

Measures the average magnitude of prediction error.

### Root Mean Squared Error (RMSE)

Measures prediction accuracy while penalizing larger errors more heavily.

These metrics are displayed directly within the dashboard to provide transparency regarding model performance.

---

## Explainability

Machine learning models often behave as black boxes.

AQI Lens incorporates explainability techniques to identify:

- Most influential environmental features
- Relative impact of pollutants
- Factors contributing to AQI variation

This improves trust, transparency, and interpretability of model predictions.

---

# System Architecture

```text
Historical AQI Dataset
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
Feature Engineering
          │
          ▼
Model Training (KNN Regressor)
          │
          ▼
Model Evaluation
          │
          ▼
Explainability Analysis
          │
          ▼
JSON Export
          │
          ▼
Interactive Dashboard
```

# Technology Stack

## Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn

## Data Visualization

- Matplotlib
- JSON-based model export

## Frontend

- HTML5
- CSS3
- JavaScript
- Chart.js

## Deployment

- GitHub
- Vercel

---

# Dashboard Modules

## Predict

Generate AQI predictions and view air-quality classifications.

## Compare

Compare AQI trends across cities using interactive visualizations.

## Explain

Understand model behavior through explainability insights.

## About

Learn about the project's objectives, methodology, and environmental relevance.

---

# Future Enhancements

- Real-time weather integration
- Advanced time-series forecasting
- Deep learning-based AQI prediction
- Expanded city coverage
- Mobile-first optimization
- Automated AQI alerts

---

# Authors

### Kevin Shiju George
B.Tech Computer Science Engineering

---

## Acknowledgements

This project was developed as an exploration of machine learning, explainable AI, and environmental analytics, demonstrating how data-driven approaches can be used to better understand and predict urban air quality.
