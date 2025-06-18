import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load or train model (simplified for demo)
# Assume model and scaler are pre-trained and loaded
# For simplicity, we'll reuse the trained model code here
np.random.seed(42)
n_samples = 1000
data = {
    'diet': np.random.choice(['vegan', 'vegetarian', 'omnivore'], n_samples),
    'transport_mode': np.random.choice(['car', 'public', 'walking'], n_samples),
    'transport_distance_km': np.random.uniform(0, 100, n_samples),
    'electricity_kwh': np.random.uniform(10, 100, n_samples)
}
df = pd.DataFrame(data)
diet_co2 = {'vegan': 0.5, 'vegetarian': 1.0, 'omnivore': 2.0}
transport_co2 = {'car': 0.2, 'public': 0.1, 'walking': 0.0}
df['diet_co2'] = df['diet'].map(diet_co2)
df['transport_co2_factor'] = df['transport_mode'].map(transport_co2)
df['co2_kg'] = df['diet_co2'] * 7 + df['transport_co2_factor'] * df['transport_distance_km'] + 0.5 * df['electricity_kwh']
X = df[['diet_co2', 'transport_distance_km', 'transport_co2_factor', 'electricity_kwh']]
y = df['co2_kg']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)

# Streamlit UI
st.title("Carbon Footprint Estimator")
diet = st.selectbox("Diet", ['vegan', 'vegetarian', 'omnivore'])
transport_mode = st.selectbox("Transport Mode", ['car', 'public', 'walking'])
transport_distance = st.slider("Weekly Transport Distance (km)", 0, 100, 50)
electricity_kwh = st.slider("Weekly Electricity Usage (kWh)", 10, 100, 50)

if st.button("Estimate CO2"):
    input_data = pd.DataFrame({
        'diet_co2': [diet_co2[diet]],
        'transport_distance_km': [transport_distance],
        'transport_co2_factor': [transport_co2[transport_mode]],
        'electricity_kwh': [electricity_kwh]
    })
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.write(f"Estimated Weekly CO2 Emissions: {prediction:.2f} kg")