# Crop Recommendation System

A machine learning-based crop recommendation system that predicts a suitable crop based on soil and environmental conditions.

## Features

- Machine learning based crop prediction
- Uses soil nutrient information
- Uses temperature and humidity
- Uses soil pH
- Uses rainfall information
- Simple web interface
- Runs locally using Flask

## Dataset

The project uses a crop recommendation dataset containing soil and environmental parameters for different crops.

### Input Features

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

### Target

The target variable is the crop label.

## Machine Learning

The project compares multiple classification algorithms:

- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine

The trained model is saved using Joblib.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- Joblib

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL