Carbon Footprint Estimator
This project implements a regression-based Carbon Footprint Estimator to predict weekly CO2 emissions (kg/week) based on lifestyle inputs: diet, transport, and electricity usage. The model is built using Python and Scikit-Learn, with an optional Streamlit interface for user interaction.
Dataset Used

Synthetic Dataset: Generated 1000 samples with the following features:
Diet: Categorical (vegan, vegetarian, omnivore) mapped to CO2 factors (0.5, 1.0, 2.0 kg/day).
Transport: Weekly distance (0-100 km) and mode (car: 0.2 kg/km, public: 0.1 kg/km, walking: 0.0 kg/km).
Electricity Usage: Weekly consumption (10-100 kWh) with a CO2 factor of 0.5 kg/kWh.
Target: Weekly CO2 emissions (kg), calculated as diet_co2 * 7 + transport_co2_factor * distance + 0.5 * electricity_kwh.



Approach Summary

Data Preparation: Created a synthetic dataset with realistic CO2 emission factors.
Preprocessing: Encoded categorical variables (diet, transport mode) and scaled features using StandardScaler.
Modeling: Trained a Linear Regression model (optionally, Random Forest) to predict CO2 emissions.
Evaluation: Computed Mean Squared Error (MSE) to assess model performance.
Visualization: Generated a scatter plot of actual vs. predicted CO2 emissions.
UI (Optional): Built a Streamlit app for users to input lifestyle data and estimate CO2 emissions.
Hosting: Code can be run in Google Colab or locally; Streamlit app deployable via Streamlit Cloud.

Dependencies

Python 3.8+
scikit-learn (for modeling)
pandas (for data handling)
numpy (for numerical operations)
matplotlib (for visualization)
streamlit (for optional UI)

Install dependencies:
pip install scikit-learn pandas numpy matplotlib streamlit

Usage

Run the Jupyter notebook (carbon_estimator.ipynb) in Google Colab or locally to train the model and visualize results.
For the Streamlit app, save the app code as app.py and run:streamlit run app.py


Deploy the Streamlit app to Streamlit Cloud by connecting to a GitHub repository.