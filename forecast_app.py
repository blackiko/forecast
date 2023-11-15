import streamlit as st
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate MAPE
def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Streamlit app
def main():
    st.title("Sales Forecasting App")

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Load and preprocess data
        sales_data_df = pd.read_excel(uploaded_file)
        sales_data_df['Month/Year'] = pd.to_datetime(sales_data_df['Month/Year'], format='%m/%Y')
        data = sales_data_df.melt(id_vars=['Month/Year'], var_name='Country', value_name='Sales')
        data = data.rename(columns={'Month/Year': 'ds', 'Sales': 'y'})

        # Add COVID-19 regressor (customize as needed)
        start_of_pandemic = pd.Timestamp('2020-01-01')
        end_of_pandemic = pd.Timestamp('2020-12-31')
        data['covid_impact'] = data['ds'].apply(lambda x: 1 if start_of_pandemic <= x <= end_of_pandemic else 0)

        unique_countries = data['Country'].unique()

        # Forecasting for each country
        for country in unique_countries:
            country_data = data[data['Country'] == country]
            recent_data = country_data[-36:]  # Last 36 months

            # Prophet model (customize parameters as needed)
            model = Prophet()
            model.add_seasonality(name='yearly', period=365.25, fourier_order=5)
            model.add_regressor('covid_impact')
            model.fit(recent_data)

            future = model.make_future_dataframe(periods=1, freq='M')
            future['covid_impact'] = future['ds'].apply(lambda x: 1 if start_of_pandemic <= x <= end_of_pandemic else 0)
            forecast = model.predict(future)

            st.subheader(f"Next Month Forecast for {country}")
            st.write(forecast[['ds', 'yhat']].tail(1))

            fig = model.plot(forecast)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
