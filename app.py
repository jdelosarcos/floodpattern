import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet

# Set up Streamlit page configuration
st.set_page_config(page_title="Flood Data Analysis and Prediction", layout="wide")

# Add a title and introduction
st.title("Flood Data Analysis and Prediction App")
st.write("Upload your flood data CSV file to analyze characteristics, predict flood occurrences, and forecast water levels using various models.")

# Add a file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    # Proceed with displaying data and analysis sections
    st.subheader("Data Overview")

    # Display data characteristics
    st.write("Shape of the DataFrame:", df.shape)
    st.write("Data types:")
    st.dataframe(df.dtypes)
    st.write("Descriptive statistics:")
    st.dataframe(df.describe())

    # Display content
    st.write("First 5 rows of the data:")
    st.dataframe(df.head())

    # Add a section for data cleaning and preprocessing
    st.subheader("Data Cleaning and Preprocessing")
    st.write("Performing data cleaning and preprocessing steps...")

    @st.cache_data # Cache the cleaned data
    def clean_water_level(dataframe):
        df_cleaned = dataframe.copy()
        df_cleaned['Water Level'] = df_cleaned['Water Level'].astype(str).str.replace(' ft.', '', regex=False).str.replace(' ft', '', regex=False).str.replace(' ', '', regex=False)
        df_cleaned['Water Level'] = df_cleaned['Water Level'].str.replace('ft', '', regex=False).replace('nan', pd.NA)
        df_cleaned['Water Level'] = pd.to_numeric(df_cleaned['Water Level'], errors='coerce')
        median_water_level = df_cleaned['Water Level'].median()
        df_cleaned['Water Level'] = df_cleaned['Water Level'].fillna(median_water_level)
        return df_cleaned

    df = clean_water_level(df)
    st.write("Water Level column cleaned and missing values imputed.")

    @st.cache_data # Cache the cleaned data for families affected
    def clean_families_affected(dataframe):
        df_cleaned = dataframe.copy()
        df_cleaned['No. of Families affected'] = df_cleaned['No. of Families affected'].astype(str).str.replace(',', '', regex=False)
        df_cleaned['No. of Families affected'] = pd.to_numeric(df_cleaned['No. of Families affected'], errors='coerce')
        median_families = df_cleaned['No. of Families affected'].median()
        df_cleaned['No. of Families affected'] = df_cleaned['No. of Families affected'].fillna(median_families)
        return df_cleaned

    df = clean_families_affected(df)
    st.write("No. of Families affected column cleaned and missing values imputed.")

    @st.cache_data # Cache the cleaned data for damage columns
    def clean_damage_columns(dataframe):
        df_cleaned = dataframe.copy()
        df_cleaned['Damage Infrastructure'] = df_cleaned['Damage Infrastructure'].astype(str).str.replace(',', '', regex=False)
        df_cleaned['Damage Infrastructure'] = pd.to_numeric(df_cleaned['Damage Infrastructure'], errors='coerce')
        df_cleaned['Damage Infrastructure'] = df_cleaned['Damage Infrastructure'].fillna(0)

        df_cleaned['Damage Agriculture'] = df_cleaned['Damage Agriculture'].astype(str).str.replace(',', '', regex=False)
        df_cleaned['Damage Agriculture'] = df_cleaned['Damage Agriculture'].str.replace('422.510.5', '4225105', regex=False)
        df_cleaned['Damage Agriculture'] = pd.to_numeric(df_cleaned['Damage Agriculture'], errors='coerce')
        df_cleaned['Damage Agriculture'] = df_cleaned['Damage Agriculture'].fillna(0)

        return df_cleaned

    df = clean_damage_columns(df)
    st.write("Damage Infrastructure and Damage Agriculture columns cleaned and missing values imputed.")

    @st.cache_data # Cache the data after handling date column missing values
    def handle_date_missing_values(dataframe):
        df_cleaned = dataframe.copy()
        # Impute missing values for 'Month', 'Day', and 'Year' using backward fill
        df_cleaned['Month'] = df_cleaned['Month'].fillna(method='bfill')
        df_cleaned['Day'] = df_cleaned['Day'].fillna(method='bfill')
        df_cleaned['Year'] = df_cleaned['Year'].fillna(method='bfill')
        return df_cleaned

    df = handle_date_missing_values(df)
    st.write("Missing values in Month, Day, and Year columns imputed.")

    # Add sections for analysis and modeling based on the overall plan
    st.subheader("Flood Analysis and Modeling")

    # --- Monthly Flood Patterns ---
    st.write("Analyzing Monthly Flood Patterns...")
    df['flood_occurred'] = (df['Water Level'] > 0).astype(int)
    monthly_flood_probability = df.groupby('Month')['flood_occurred'].mean().sort_values(ascending=False)
    st.write("Monthly Flood Probabilities (Sorted):")
    st.dataframe(monthly_flood_probability)
    plt.figure(figsize=(12, 7))
    monthly_flood_probability.plot(kind='bar', color='skyblue')
    plt.title('Monthly Flood Probability')
    plt.xlabel('Month')
    plt.ylabel('Probability of Flood Occurrence')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # --- Clustering Analysis ---
    st.write("Performing Clustering Analysis...")
    @st.cache_resource # Cache the KMeans model
    def perform_clustering(dataframe):
        selected_columns_df = dataframe[['Municipality', 'Barangay', 'Flood Cause', 'Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']].copy()
        categorical_cols = selected_columns_df.select_dtypes(include='object').columns
        encoded_df = pd.get_dummies(selected_columns_df, columns=categorical_cols, dummy_na=False)
        # Determine optimal number of clusters if possible, or use a predefined number
        n_clusters = 3 # Example: Using 3 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(encoded_df)
        dataframe['Cluster'] = kmeans.labels_
        return dataframe

    df = perform_clustering(df)
    st.write("Clustering analysis performed. Cluster labels added to the data.")
    st.write("Count of instances in each cluster:")
    st.dataframe(df['Cluster'].value_counts())

    # --- Flood Severity Prediction ---
    st.write("Analyzing Flood Severity...")
    def categorize_severity(water_level):
        if water_level <= 5:
            return 'Low'
        elif 5 < water_level <= 15:
            return 'Medium'
        else:
            return 'High'
    df['Flood_Severity'] = df['Water Level'].apply(categorize_severity)
    st.write("Distribution of Flood Severity:")
    st.dataframe(df['Flood_Severity'].value_counts())
    # Note: Training a predictive model for severity would require more code,
    # including feature selection, encoding, splitting data, training, and evaluation.
    # This is a placeholder to indicate where that analysis would go.
    st.write("Further analysis and modeling for flood severity can be integrated here.")

    # --- Time Series Forecasting (SARIMA/Prophet) ---
st.write("Performing Time Series Forecasting...")
try:
    # Ensure columns exist
    required_cols = ['Year', 'Month', 'Day', 'Water Level']
    if not all(col in df.columns for col in required_cols):
        st.warning("Missing one or more required columns for time series analysis.")
    else:
        # Convert to proper date
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Day'] = pd.to_numeric(df['Day'], errors='coerce')

        month_map = {
            'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
            'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10,
            'NOVEMBER': 11, 'DECEMBER': 12, 'Unknown': 1
        }
        df['Month_Num'] = df['Month'].map(month_map)
        df['Date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month_Num'], day=df['Day']), errors='coerce')

        # Drop invalid or missing dates
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date').sort_index()

        # Resample daily
        ts_df_filled = df['Water Level'].resample('D').mean().fillna(method='ffill').fillna(method='bfill')

        st.write("Daily Average Water Level Time Series:")
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(ts_df_filled)
        ax.set_title('Daily Average Water Level Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Water Level')
        st.pyplot(fig)
        plt.close(fig)

        # --- SARIMA Model ---
        st.write("Training SARIMA Model...")
        @st.cache_resource
        def train_sarima_model(ts_data):
            optimal_sarima_order = (1, 1, 1)
            optimal_seasonal_order = (0, 0, 0, 7)
            warnings.filterwarnings("ignore")
            model = SARIMAX(ts_data, order=optimal_sarima_order,
                            seasonal_order=optimal_seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            return model.fit()

        results_sarima = train_sarima_model(ts_df_filled)
        st.success("SARIMA Model Trained Successfully.")
        st.text(results_sarima.summary())

        # --- Prophet Model ---
        st.write("Training Prophet Model...")
        @st.cache_resource
        def train_prophet_model(ts_data):
            prophet_df = ts_data.reset_index()
            prophet_df.columns = ['ds', 'y']  # Prophet expects these names
            model = Prophet()
            model.fit(prophet_df)
            return model, prophet_df

        model_prophet, prophet_df_for_future = train_prophet_model(ts_df_filled)
        st.success("Prophet Model Trained Successfully.")

        # --- Compare Models ---
        st.subheader("Model Comparison and Forecasting")

        # Fitted values
        fitted_sarima = results_sarima.fittedvalues
        forecast_prophet_hist = model_prophet.predict(prophet_df_for_future[['ds']])
        fitted_prophet = forecast_prophet_hist.set_index('ds')['yhat'].reindex(ts_df_filled.index)

        # Metrics
        rmse_sarima = np.sqrt(mean_squared_error(ts_df_filled, fitted_sarima))
        mae_sarima = mean_absolute_error(ts_df_filled, fitted_sarima)
        rmse_prophet = np.sqrt(mean_squared_error(ts_df_filled, fitted_prophet))
        mae_prophet = mean_absolute_error(ts_df_filled, fitted_prophet)

        perf_df = pd.DataFrame({
            'Model': ['SARIMA', 'Prophet'],
            'RMSE': [rmse_sarima, rmse_prophet],
            'MAE': [mae_sarima, mae_prophet]
        })
        st.dataframe(perf_df)

        best_model = perf_df.loc[perf_df['RMSE'].idxmin(), 'Model']
        st.write(f"✅ Best Model Based on RMSE: **{best_model}**")

        # --- Forecast Future 30 Days ---
        steps_ahead = 30
        last_date = ts_df_filled.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps_ahead, freq='D')

        if best_model == 'SARIMA':
            forecast = results_sarima.get_forecast(steps=steps_ahead)
            future_forecast = forecast.predicted_mean
        else:
            future_df = model_prophet.make_future_dataframe(periods=steps_ahead)
            future_forecast_prophet = model_prophet.predict(future_df)
            future_forecast = future_forecast_prophet.set_index('ds')['yhat'].tail(steps_ahead)

        # --- Plot Forecast ---
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(ts_df_filled.index, ts_df_filled, label='Historical Data')
        ax.plot(future_forecast.index, future_forecast, color='red', label='Future Forecast')
        ax.set_title(f"{best_model} Model Forecast (Next 30 Days)")
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Water Level')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

except Exception as e:
    st.error(f"❌ Time Series Analysis Error: {e}")

