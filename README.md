# Solubility Prediction Web App

A machine learning web app that predicts the solubility (LogS) of molecules.

## Features
- Predicts molecular solubility from chemical descriptors
- Built with a trained machine learning model
- Simple and interactive UI

## How to use
1. Enter molecular descriptors in the input fields
2. Click **Predict** to get the solubility value
3. Results are displayed instantly

## Example input
| Descriptor | Value |
|-----------|-------|
| MolLogP | 2.5 |
| MolWt | 167.85 |
| NumRotatableBonds | 0 |
| AromaticProportion | 0.0 |

## Installation
pip install -r requirements.txt

## Run the app
streamlit run solubility-app.py

## Built with
- Python
- Streamlit
- Pandas
- Scikit-learn
- Pickle
