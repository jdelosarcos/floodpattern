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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    @st.cache_resource # Cache the clustering results and transformation objects
    def perform_clustering(dataframe, n_clusters=3):
        # Select relevant columns for clustering
        selected_columns_df = dataframe[['Municipality', 'Barangay', 'Flood Cause', 'Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']].copy()
        categorical_cols = selected_columns_df.select_dtypes(include='object').columns
        encoded_df = pd.get_dummies(selected_columns_df, columns=categorical_cols, dummy_na=False)

        # Standardize
        scaler = StandardScaler()
        encoded_scaled = scaler.fit_transform(encoded_df)

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(encoded_scaled)

        # Add cluster labels to a copy of the original dataframe
        df_with_cluster = dataframe.copy()
        df_with_cluster['Cluster'] = labels

        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(encoded_scaled)
        pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'], index=encoded_df.index)

        # Transform centroids to PCA space for plotting
        centroids_pca = pca.transform(kmeans.cluster_centers_)

        return df_with_cluster, pca_df, centroids_pca, kmeans, encoded_df.columns.tolist()

    # run clustering (choose number of clusters interactively)
    n_clusters = st.sidebar.slider("Number of clusters (KMeans)", min_value=2, max_value=8, value=3)
    df, pca_df, centroids_pca, kmeans_model, encoded_cols = perform_clustering(df, n_clusters=n_clusters)
    st.write("Clustering analysis performed. Cluster labels added to the data.")
    st.write("Count of instances in each cluster:")
    st.dataframe(df['Cluster'].value_counts())

    # Plot clustering result in 2D (PCA)
    st.subheader("Clustering Visualization (PCA 2D projection)")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=df['Cluster'], cmap='tab10', alpha=0.7)
    # plot centroids
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='black', label='Centroids')
    ax.set_title('PCA projection of clustered instances')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(*scatter.legend_elements(), title="Clusters")
    st.pyplot(fig)
    plt.close(fig)

    # Provide UI to display a specific cluster
    st.subheader("Explore a Specific Cluster")
    available_clusters = sorted(df['Cluster'].unique().tolist())
    selected_cluster = st.selectbox("Select cluster to view", available_clusters)
    cluster_df = df[df['Cluster'] == selected_cluster]
    st.write(f"Rows in Cluster {selected_cluster}: {len(cluster_df)}")
    st.dataframe(cluster_df.head(200))  # show up to first 200 rows

    # Additional cluster-specific plots
    st.write(f"Water Level distribution for Cluster {selected_cluster}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cluster_df['Water Level'], bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Water Level')
    ax.set_ylabel('Count')
    ax.set_title(f'Water Level Histogram - Cluster {selected_cluster}')
    st.pyplot(fig)
    plt.close(fig)

    # Show cluster points highlighted on PCA scatter
    st.write("PCA scatter with selected cluster highlighted")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pca_df['PC1'], pca_df['PC2'], color='lightgray', alpha=0.5, label='Other Clusters')
    sel_idx = cluster_df.index
    # Ensure index alignment between pca_df and df
    sel_pca = pca_df.loc[sel_idx]
    ax.scatter(sel_pca['PC1'], sel_pca['PC2'], color='red', alpha=0.8, label=f'Cluster {selected_cluster}')
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='black', label='Centroids')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA projection - Cluster {selected_cluster} highlighted')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

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
        required_cols = ['Year', 'Month', 'Day', 'Water Level']
        if not all(col in df.columns for col in required_cols):
            st.warning("Missing one or more required columns for time series analysis.")
        else:
            # --- Clean and prepare date components ---
            df_ts = df.copy()

            # Convert Year and Day to numeric safely
            df_ts['Year'] = pd.to_numeric(df_ts['Year'], errors='coerce')
            df_ts['Day'] = pd.to_numeric(df_ts['Day'], errors='coerce')

            # Map month names or numbers safely
            month_map = {
                'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
                'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10,
                'NOVEMBER': 11, 'DECEMBER': 12, 'Unknown': 1
            }

            # Handle both text and numeric month formats
            df_ts['Month_Num'] = df_ts['Month'].apply(lambda x: month_map.get(str(x).strip().upper(), pd.to_numeric(x, errors='coerce')))

            # Drop rows with any missing date component
            df_ts = df_ts.dropna(subset=['Year', 'Month_Num', 'Day', 'Water Level'])

            # Convert columns to int safely (only valid rows remain)
            df_ts['Year'] = df_ts['Year'].astype(int)
            df_ts['Month_Num'] = df_ts['Month_Num'].astype(int)
            df_ts['Day'] = df_ts['Day'].astype(int)

            # Create valid datetime column
            df_ts['Date'] = pd.to_datetime(dict(year=df_ts['Year'], month=df_ts['Month_Num'], day=df_ts['Day']), errors='coerce')

            # Drop invalid or missing dates
            df_ts = df_ts.dropna(subset=['Date'])
            df_ts = df_ts.set_index('Date').sort_index()

            # --- Resample daily average ---
            ts_df_filled = df_ts['Water Level'].resample('D').mean().fillna(method='ffill').fillna(method='bfill')

            st.write("📈 Daily Average Water Level Time Series:")
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(ts_df_filled, label='Observed Water Level')
            ax.set_title('Daily Average Water Level Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Water Level')
            ax.legend()
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
                return model.fit(disp=False)

            results_sarima = train_sarima_model(ts_df_filled)
            st.success("✅ SARIMA Model Trained Successfully.")
            st.text(results_sarima.summary())

            # --- Prophet Model ---
            st.write("Training Prophet Model...")
            @st.cache_resource
            def train_prophet_model(ts_data):
                prophet_df = ts_data.reset_index()
                prophet_df.columns = ['ds', 'y']  # Prophet expects these column names
                model = Prophet()
                model.fit(prophet_df)
                return model, prophet_df

            model_prophet, prophet_df_for_future = train_prophet_model(ts_df_filled)
            st.success("✅ Prophet Model Trained Successfully.")

            # --- Compare Models ---
            st.subheader("Model Comparison and Forecasting")

            # Fitted values
            # SARIMA fitted values (in-sample prediction)
            sarima_pred_insample = results_sarima.get_prediction(start=ts_df_filled.index[0], end=ts_df_filled.index[-1])
            fitted_sarima = sarima_pred_insample.predicted_mean
            sarima_ci = sarima_pred_insample.conf_int()

            # Prophet in-sample prediction
            forecast_prophet_hist = model_prophet.predict(prophet_df_for_future[['ds']])
            fitted_prophet = forecast_prophet_hist.set_index('ds')['yhat'].reindex(ts_df_filled.index)
            prophet_ci_lower = forecast_prophet_hist.set_index('ds')['yhat_lower'].reindex(ts_df_filled.index)
            prophet_ci_upper = forecast_prophet_hist.set_index('ds')['yhat_upper'].reindex(ts_df_filled.index)

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
            st.write(f"🏆 Best Model Based on RMSE: **{best_model}**")

            # --- Plot SARIMA results (in-sample fitted + forecast) ---
            st.subheader("SARIMA: Observed vs Fitted (in-sample) and Forecast")
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(ts_df_filled.index, ts_df_filled, label='Observed', color='black')
            ax.plot(fitted_sarima.index, fitted_sarima, label='SARIMA Fitted', color='blue')
            # plot confidence interval
            ax.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='blue', alpha=0.2, label='SARIMA 95% CI')

            # Forecast next 30 days from SARIMA
            steps_ahead = 30
            sarima_forecast = results_sarima.get_forecast(steps=steps_ahead)
            sarima_forecast_mean = sarima_forecast.predicted_mean
            sarima_forecast_ci = sarima_forecast.conf_int()

            # Only plot SARIMA forecast if forecasting for SARIMA (we always show this plot)
            # Prepare x values
            ax.plot(sarima_forecast_mean.index, sarima_forecast_mean, color='red', linestyle='--', label='SARIMA Forecast')
            ax.fill_between(sarima_forecast_ci.index, sarima_forecast_ci.iloc[:, 0], sarima_forecast_ci.iloc[:, 1], color='red', alpha=0.15, label='SARIMA Forecast 95% CI')

            ax.set_title('SARIMA - Observed vs Fitted and 30-day Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Level')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # --- Plot Prophet results (in-sample fitted + forecast) ---
            st.subheader("Prophet: Observed vs Fitted (in-sample) and Forecast")
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(ts_df_filled.index, ts_df_filled, label='Observed', color='black')
            ax.plot(fitted_prophet.index, fitted_prophet, label='Prophet Fitted', color='green')

            # Prophet CI (in-sample)
            ax.fill_between(fitted_prophet.index, prophet_ci_lower, prophet_ci_upper, color='green', alpha=0.2, label='Prophet 95% CI')

            # Forecast future with Prophet
            future_df = model_prophet.make_future_dataframe(periods=steps_ahead)
            future_forecast_prophet = model_prophet.predict(future_df)
            future_forecast = future_forecast_prophet.set_index('ds')['yhat'].tail(steps_ahead)
            future_lower = future_forecast_prophet.set_index('ds')['yhat_lower'].tail(steps_ahead)
            future_upper = future_forecast_prophet.set_index('ds')['yhat_upper'].tail(steps_ahead)

            ax.plot(future_forecast.index, future_forecast, color='orange', linestyle='--', label='Prophet Forecast')
            ax.fill_between(future_forecast.index, future_lower, future_upper, color='orange', alpha=0.15, label='Prophet Forecast 95% CI')

            ax.set_title('Prophet - Observed vs Fitted and 30-day Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Level')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # --- Combined forecast plot (historical + forecasts from both models) ---
            st.subheader("Combined Forecasts (Historical + SARIMA & Prophet 30-day forecasts)")
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(ts_df_filled.index, ts_df_filled, label='Observed', color='black')

            # Plot SARIMA forecast
            ax.plot(sarima_forecast_mean.index, sarima_forecast_mean, color='red', linestyle='--', label='SARIMA Forecast')

            # Plot Prophet forecast
            ax.plot(future_forecast.index, future_forecast, color='orange', linestyle='--', label='Prophet Forecast')

            ax.set_title('Historical Data and 30-day Forecasts from SARIMA & Prophet')
            ax.set_xlabel('Date')
            ax.set_ylabel('Water Level')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(f"❌ Could not perform time series analysis and forecasting: {e}")
