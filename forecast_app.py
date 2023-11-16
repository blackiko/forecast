import streamlit as st
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Function to calculate MAPE
def mape(y_true, y_pred): 
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Parameters for each country
country_params = {
    'ID': {'changepoint_prior_scale': 1.59, 'seasonality_prior_scale': 0.99},
    'MY': {'changepoint_prior_scale': 9.75, 'seasonality_prior_scale': 0.084},
    'PH': {'changepoint_prior_scale': 8.66, 'seasonality_prior_scale': 0.55},
    'SG': {'changepoint_prior_scale': 6.53, 'seasonality_prior_scale': 0.54},
    'TH': {'changepoint_prior_scale': 0.80, 'seasonality_prior_scale': 0.55},
    'VN': {'changepoint_prior_scale': 2.65, 'seasonality_prior_scale': 0.12}
}

# Streamlit app
def main():
    st.title("Sales Forecasting App")
    st.write("Upload your sales data in Excel format. Adjust forecasting parameters as needed.")
    st.write("The Excel file should contain two or more columns. First Column named 'Month/Year' with dates in MM/YYYY format. Second column onwards named '{country name}' with sales data of respective countries.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Load and preprocess data
        sales_data_df = pd.read_excel(uploaded_file)
        sales_data_df['Month/Year'] = pd.to_datetime(sales_data_df['Month/Year'], format='%m/%Y')
        data = sales_data_df.melt(id_vars=['Month/Year'], var_name='Country', value_name='Sales')
        data = data.rename(columns={'Month/Year': 'ds', 'Sales': 'y'})

        # Handle missing values
        data.replace(0, np.nan, inplace=True)

        # Add COVID-19 regressor
        start_of_pandemic = pd.Timestamp('2020-01-01')
        end_of_pandemic = pd.Timestamp('2020-12-31')
        data['covid_impact'] = data['ds'].apply(lambda x: 1 if start_of_pandemic <= x <= end_of_pandemic else 0)

        unique_countries = data['Country'].unique()

        # Customizable Date Range
        min_date, max_date = data['ds'].dt.date.min(), data['ds'].dt.date.max()
        start_date, end_date = st.slider("Select Date Range for Historical Data", min_date, max_date, (min_date, max_date))
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)

        # Slider for forecast period
        forecast_period = st.slider("Select Forecast Period (Months)", 1, 12, 1)

        # Allow users to manually adjust parameters
        st.sidebar.title("Forecasting Parameters")

        # Explanation of Parameters
        st.sidebar.title("Parameter Explanations")
        st.sidebar.write("Changepoint Prior Scale: Controls how sensitive the model is to changes in the trend. Higher values allow the model to fit rapid changes more closely.")
        st.sidebar.write("Seasonality Prior Scale: Controls the flexibility of the seasonality component. Higher values allow the model to fit seasonal variations more closely.")
        st.sidebar.write("Fourier Order: Determines the complexity of the seasonality model. Higher values capture more seasonal fluctuations but can lead to overfitting.")
        
        manual_param_input = st.sidebar.checkbox("Manually adjust parameters", False)

        # Forecasting for each country
        for country in unique_countries:
            country_data = data[(data['Country'] == country) & (data['ds'] >= start_date) & (data['ds'] <= end_date)]

            # Default parameters for the country
            params = country_params.get(country, {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.05})
            changepoint_prior_scale = params['changepoint_prior_scale']
            seasonality_prior_scale = params['seasonality_prior_scale']
            fourier_order = params.get('fourier_order', 1)

            if manual_param_input:
                changepoint_prior_scale = st.sidebar.slider(f"Changepoint Prior Scale for {country}", 0.001, 10.0, float(changepoint_prior_scale), step=0.01)
                seasonality_prior_scale = st.sidebar.slider(f"Seasonality Prior Scale for {country}", 0.001, 10.0, float(seasonality_prior_scale), step=0.01)
                fourier_order = st.sidebar.slider(f"Fourier Order for {country}", 1, 12, 1)

            # Prophet model with parameters
            model = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_prior_scale=seasonality_prior_scale)
            model.add_seasonality(name='yearly', period=365.25, fourier_order=fourier_order)
            model.add_regressor('covid_impact')
            model.fit(country_data)

            future = model.make_future_dataframe(periods=forecast_period, freq='M')
            future['covid_impact'] = future['ds'].apply(lambda x: 1 if start_of_pandemic <= x <= end_of_pandemic else 0)
            forecast = model.predict(future)

            # Align actuals and predictions for MAPE calculation
            actuals = country_data.set_index('ds')['y'].dropna()
            predictions = forecast.set_index('ds')['yhat']
            overlapping_dates = actuals.index.intersection(predictions.index)
            mape_value = mape(actuals.loc[overlapping_dates], predictions.loc[overlapping_dates])

            st.subheader(f"Forecast for {country} for Next {forecast_period} Months")
            st.write(f"MAPE: {mape_value:.2f}%" if not np.isnan(mape_value) else "MAPE: N/A (Insufficient Data)")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))

            # Historical data visualization
            fig = px.line(country_data, x='ds', y='y', title=f"Sales Forecast vs Historical Data for {country}", 
                        labels={'ds': 'Date', 'y': 'Sales'}, height=500)
            fig.update_traces(name='Actual Sales', line_color='lightblue', line_width=3)  # Light blue color for historical data
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', 
                            line=dict(color='red', width=3))  # Red color for forecast line
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', 
                            name='Upper Confidence Interval', line=dict(width=0), 
                            fillcolor='rgba(255,0,0,0.3)')  # Light red fill for upper confidence
            fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', 
                            name='Lower Confidence Interval', line=dict(width=0), 
                            fillcolor='rgba(255,0,0,0.3)')  # Light red fill for lower confidence

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                font_color='white',  # White font for better visibility
                legend_title_text='Legend'  # Legend title
            )

            st.plotly_chart(fig, use_container_width=True)

            # Option to download the forecast data
            st.download_button(
                label=f"Download {country} Forecast Data",
                data=forecast.to_csv().encode('utf-8'),
                file_name=f'{country}_forecast.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
