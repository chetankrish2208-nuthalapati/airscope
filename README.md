# Readme.md

#Airscope

## Multimodal Air-Quality and Environmental Risk Prototype

AirScope is an educational environmental-AI prototype that combines pollutant measurements with environmental-image context to estimate regional pollution risk.

## Problem

Air-quality dashboards often show a number without explaining the surrounding environmental conditions. AirScope is designed to provide a simple, explainable risk category for educational and community-awareness use.

## Current features

- Uses PM2.5, PM10, NO2, SO2, CO, and O3 measurements.
- Includes temperature, humidity, and wind speed.
- Classifies risk as Low, Moderate, or High.
- Uses a Random Forest classifier.
- Provides model confidence.
- Supports location display using a Folium map.
- Allows environmental image upload for visual context.
- Includes uncertainty and safety limitations.

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Folium
- Streamlit-Folium
- Pillow
- Joblib

## Dataset

The project uses a global air-quality dataset containing pollutant and weather measurements.

The dataset does not contain an official AQI column. Therefore, the prototype creates an educational pollution-risk label from pollutant values using a weighted pollution score.

## Method

The model uses:

- PM2.5
- PM10
- NO2
- SO2
- CO
- O3
- Temperature
- Humidity
- Wind Speed

A Random Forest classifier predicts Low, Moderate, or High pollution risk.

## Results

The current model was trained using 10,000 rows.

Accuracy on the test split: 95.1%

High-risk examples were less common than Low and Moderate examples, so accuracy should not be interpreted as official pollution-prediction performance.

## Responsible AI limitations

- This is not an official AQI service.
- Satellite or environmental images do not directly measure ground-level AQI.
- The image component provides contextual visual evidence only.
- The system should not be used for medical, emergency, or official school-closure decisions.
- Users should check official local air-quality information.

## Future work

- Add real satellite-derived aerosol data.
- Use time-based AQI forecasting.
- Improve class balance.
- Test on geographically separate cities.
- Add uncertainty calibration.
- Compare predictions with official monitoring-station data.

## Author

Nuthalapati Chetan Krishna