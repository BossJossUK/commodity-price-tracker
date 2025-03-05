import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
import plotly.graph_objects as go
import time
import numpy as np
from prophet import Prophet
import traceback

# Page configuration
st.set_page_config(
    page_title="Metal & Oil Price Tracker",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Metal & Oil Price Tracker")
st.markdown("Track historical prices for Copper, Aluminium, Steel, and Oil")

# Function to generate forecast using Prophet
def generate_forecast(data, periods=30, commodity_name="Commodity"):
    """
    Generate forecast using Prophet
    
    Args:
        data: DataFrame with 'Date' and 'Price' columns
        periods: Number of days to forecast
        commodity_name: Name of the commodity for display
        
    Returns:
        Tuple of (forecast DataFrame, Prophet model)
    """
    # Prophet requires columns named 'ds' and 'y'
    df_prophet = data.rename(columns={'Date': 'ds', 'Price': 'y'})
    
    # Create and fit the model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05  # Flexibility of the trend
    )
    
    # Add additional regressors if available
    if 'Open' in data.columns and 'High' in data.columns and 'Low' in data.columns:
        # Create lagged variables for technical indicators
        df_prophet['price_lag1'] = df_prophet['y'].shift(1)
        df_prophet['price_lag2'] = df_prophet['y'].shift(2)
        df_prophet['price_lag3'] = df_prophet['y'].shift(3)
        
        # Drop NaN values
        df_prophet = df_prophet.dropna()
        
        # Add regressors
        model.add_regressor('price_lag1')
        model.add_regressor('price_lag2')
        model.add_regressor('price_lag3')
    
    # Fit the model
    model.fit(df_prophet)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Add regressor values to future dataframe if they exist
    if 'price_lag1' in df_prophet.columns:
        # Get the last values for lagged variables
        last_values = df_prophet[['y', 'price_lag1', 'price_lag2']].iloc[-1].values
        
        # Fill in known values
        future.loc[future['ds'].isin(df_prophet['ds']), 'price_lag1'] = df_prophet['price_lag1'].values
        future.loc[future['ds'].isin(df_prophet['ds']), 'price_lag2'] = df_prophet['price_lag2'].values
        future.loc[future['ds'].isin(df_prophet['ds']), 'price_lag3'] = df_prophet['price_lag3'].values
        
        # For future dates, use the forecasted values (simplified approach)
        for i in range(len(df_prophet), len(future)):
            if i-1 >= len(df_prophet):
                # Use previous predictions for future dates
                future.loc[i, 'price_lag1'] = model.predict(future.iloc[[i-1]])['yhat'].values[0]
                if i-2 >= len(df_prophet):
                    future.loc[i, 'price_lag2'] = model.predict(future.iloc[[i-2]])['yhat'].values[0]
                else:
                    future.loc[i, 'price_lag2'] = df_prophet['y'].iloc[-1]
                if i-3 >= len(df_prophet):
                    future.loc[i, 'price_lag3'] = model.predict(future.iloc[[i-3]])['yhat'].values[0]
                else:
                    future.loc[i, 'price_lag3'] = df_prophet['y'].iloc[-2]
            else:
                # Use actual values for the first prediction
                future.loc[i, 'price_lag1'] = df_prophet['y'].iloc[-1]
                future.loc[i, 'price_lag2'] = df_prophet['price_lag1'].iloc[-1]
                future.loc[i, 'price_lag3'] = df_prophet['price_lag2'].iloc[-1]
    
    # Generate forecast
    forecast = model.predict(future)
    
    return forecast, model

# Function to load all csv files from data directory
@st.cache_data(show_spinner=False)
def load_data():
    # Path is relative to where the app is run
    data_path = "data"
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    # Dictionary to store dataframes by commodity name
    commodities = {}
    
    # Prioritize specific commodity files we know exist
    commodity_patterns = {
        "copper": "*copper*data.csv",
        "aluminium": "*aluminium*data.csv",
        "steel": "*steel*data.csv",
        "oil": "*oil*data.csv"
    }
    
    # First load our known commodities
    for commodity, pattern in commodity_patterns.items():
        matching_files = glob.glob(os.path.join(data_path, pattern))
        
        if matching_files:
            # Skip raw data files
            matching_files = [f for f in matching_files if "_raw" not in f]
            
            # Take the first match for each timeframe
            daily_file = [f for f in matching_files if "daily" in f.lower()]
            weekly_file = [f for f in matching_files if "weekly" in f.lower()]
            monthly_file = [f for f in matching_files if "monthly" in f.lower()]
            
            try:
                # Prefer daily data, then weekly, then monthly
                if daily_file:
                    df = pd.read_csv(daily_file[0], parse_dates=['Date'])
                    commodities[f"{commodity}_daily"] = df
                
                if weekly_file:
                    df = pd.read_csv(weekly_file[0], parse_dates=['Date'])
                    commodities[f"{commodity}_weekly"] = df
                
                if monthly_file:
                    df = pd.read_csv(monthly_file[0], parse_dates=['Date'])
                    commodities[f"{commodity}_monthly"] = df
                
                # Set a default for each commodity (daily preferred)
                if daily_file:
                    commodities[commodity] = commodities[f"{commodity}_daily"]
                elif weekly_file:
                    commodities[commodity] = commodities[f"{commodity}_weekly"]
                elif monthly_file:
                    commodities[commodity] = commodities[f"{commodity}_monthly"]
                    
            except Exception as e:
                # Skip this commodity if there's an error
                continue
    
    # Also try to load consolidated data
    consolidated_file = os.path.join(data_path, "consolidated_weekly_prices.csv")
    if os.path.exists(consolidated_file):
        try:
            df = pd.read_csv(consolidated_file, parse_dates=['Date'])
            commodities["consolidated_weekly"] = df
        except Exception:
            pass
    
    return commodities

# Load data with timeout
start_time = time.time()
with st.spinner("Loading commodity data..."):
    commodity_data = load_data()
    if time.time() - start_time > 10:  # 10 second timeout
        st.error("Data loading timed out. Please refresh the page.")
        st.stop()

# Check if any data was loaded
if not commodity_data:
    st.warning("No commodity data found. Please make sure CSV files are in the data directory.")
    st.stop()

# Sidebar for selecting commodities
st.sidebar.header("Settings")

# Add version info in sidebar
st.sidebar.info("Last data update: March 2025")

# Commodity selection
available_commodities = list(commodity_data.keys())
# Filter out timeframe-specific entries for cleaner dropdown
display_commodities = [c for c in available_commodities if not (c.endswith('_daily') or c.endswith('_weekly') or c.endswith('_monthly')) and c != 'consolidated_weekly']

selected_commodity = st.sidebar.selectbox(
    "Select Commodity", 
    options=display_commodities
)

# Timeframe selection
if selected_commodity in commodity_data:
    # Check if timeframe variants exist
    timeframes = []
    if f"{selected_commodity}_daily" in commodity_data:
        timeframes.append("Daily")
    if f"{selected_commodity}_weekly" in commodity_data:
        timeframes.append("Weekly")
    if f"{selected_commodity}_monthly" in commodity_data:
        timeframes.append("Monthly")
    
    if timeframes:
        selected_timeframe = st.sidebar.radio(
            "Select Timeframe",
            options=timeframes,
            horizontal=True
        )
        
        # Use the appropriate dataframe based on timeframe
        if selected_timeframe == "Daily" and f"{selected_commodity}_daily" in commodity_data:
            df = commodity_data[f"{selected_commodity}_daily"]
        elif selected_timeframe == "Weekly" and f"{selected_commodity}_weekly" in commodity_data:
            df = commodity_data[f"{selected_commodity}_weekly"]
        elif selected_timeframe == "Monthly" and f"{selected_commodity}_monthly" in commodity_data:
            df = commodity_data[f"{selected_commodity}_monthly"]
        else:
            df = commodity_data[selected_commodity]
    else:
        df = commodity_data[selected_commodity]
    
    # Date range selection
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Handle single date selection
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range[0]
        end_date = date_range[0]
    
    # Filter data based on date range
    filtered_df = df[(df['Date'].dt.date >= start_date) & 
                      (df['Date'].dt.date <= end_date)]
    
    # Main content area - split into tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Charts", "📊 Comparison", "🔮 Prediction", "🗃️ Data Explorer"])
    
    with tab1:
        st.header(f"{selected_commodity.capitalize()} Price Trends")
        
        # Plot options
        chart_type = st.radio(
            "Chart Type", 
            options=["Line Chart", "Candlestick Chart", "Area Chart"],
            horizontal=True
        )
        
        if chart_type == "Line Chart":
            # Use Plotly for better interactive charts
            price_col = 'Price'
            fig = px.line(
                filtered_df, 
                x='Date', 
                y=price_col,
                title=f"{selected_commodity.capitalize()} Price Trend",
                labels={'Date': 'Date', price_col: 'Price'},
                template='plotly_white'
            )
            fig.update_traces(line=dict(width=2))
            fig.update_layout(hovermode="x unified")
            
        elif chart_type == "Candlestick Chart" and all(col in filtered_df.columns for col in ['Open', 'High', 'Low']):
            # Use Plotly's candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=filtered_df['Date'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Price'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.update_layout(
                title=f"{selected_commodity.capitalize()} Candlestick Chart",
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white'
            )
        else:  # Area chart or fallback if candlestick data not available
            price_col = 'Price'
            fig = px.area(
                filtered_df, 
                x='Date', 
                y=price_col,
                title=f"{selected_commodity.capitalize()} Price Trend",
                labels={'Date': 'Date', price_col: 'Price'},
                template='plotly_white'
            )
            fig.update_layout(hovermode="x unified")
        
        # Display the plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.subheader("Price Statistics")
        price_col = 'Price'
        stats = filtered_df[price_col].describe()
        
        # Format the stats display
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Price", f"${stats['mean']:.2f}")
        col2.metric("Min Price", f"${stats['min']:.2f}")
        col3.metric("Max Price", f"${stats['max']:.2f}")
        col4.metric("Price Volatility", f"${stats['std']:.2f}")
        
    with tab2:
        st.header("Commodity Price Comparison")
        
        # Try to load the consolidated data
        if "consolidated_weekly" in commodity_data:
            consolidated_df = commodity_data["consolidated_weekly"]
            
            # Filter by date range
            consolidated_filtered = consolidated_df[
                (consolidated_df['Date'].dt.date >= start_date) & 
                (consolidated_df['Date'].dt.date <= end_date)
            ]
            
            # Select which commodities to display
            commodities_to_show = st.multiselect(
                "Select commodities to compare",
                options=['Price Aluminium', 'Price Copper', 'Price Steel', 'Price Oil'],
                default=['Price Aluminium', 'Price Copper', 'Price Steel', 'Price Oil']
            )
            
            if commodities_to_show and not consolidated_filtered.empty:
                # Create comparison chart
                fig = px.line(
                    consolidated_filtered, 
                    x='Date', 
                    y=commodities_to_show,
                    title="Commodity Price Comparison",
                    labels={'value': 'Price', 'variable': 'Commodity'},
                    template='plotly_white'
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add normalized comparison option
                if st.checkbox("Show normalized comparison (percentage change)"):
                    # Normalize to the first value (100%)
                    normalized_df = consolidated_filtered.copy()
                    for column in commodities_to_show:
                        if not normalized_df.empty and column in normalized_df.columns:
                            first_value = normalized_df[column].iloc[0]
                            if first_value != 0:  # Avoid division by zero
                                normalized_df[f"{column} (%)"] = normalized_df[column] / first_value * 100
                    
                    normalized_columns = [f"{col} (%)" for col in commodities_to_show if f"{col} (%)" in normalized_df.columns]
                    
                    if normalized_columns:
                        fig = px.line(
                            normalized_df, 
                            x='Date', 
                            y=normalized_columns,
                            title="Normalized Price Comparison (First date = 100%)",
                            labels={'value': 'Percentage Change', 'variable': 'Commodity'},
                            template='plotly_white'
                        )
                        fig.update_layout(hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
            elif consolidated_filtered.empty:
                st.warning("No data available for the selected date range")
            else:
                st.warning("Please select at least one commodity to display")
        else:
            st.warning("Consolidated price data file not found.")

    with tab3:
        st.header(f"{selected_commodity.capitalize()} Price Prediction")
        
        # Create two columns for prediction settings
        col1, col2 = st.columns(2)
        
        with col1:
            # Number of days to forecast
            forecast_days = st.slider("Forecast days", min_value=7, max_value=90, value=30, step=7)
            
        with col2:
            # Include uncertainty intervals
            show_uncertainty = st.checkbox("Show uncertainty intervals", value=True)
            # Include components
            show_components = st.checkbox("Show forecast components", value=True)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    # Use a copy of the full dataset for forecasting (not just filtered data)
                    forecast_df = df.copy()
                    
                    # Generate forecast
                    forecast, model = generate_forecast(
                        forecast_df, 
                        periods=forecast_days,
                        commodity_name=selected_commodity
                    )
                    
                    # Create plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Price'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    # Add prediction
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red')
                    ))
                    
                    # Add prediction intervals if selected
                    if show_uncertainty:
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_lower'],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(255, 0, 0, 0.1)',
                            line=dict(width=0),
                            name='Confidence Interval'
                        ))
                    
                    fig.update_layout(
                        title=f"{selected_commodity.capitalize()} Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        hovermode="x unified",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast components if selected
                    if show_components:
                        st.subheader("Forecast Components")
                        try:
                            # Make a copy of the forecast for component plotting
                            forecast_comp = forecast.copy()
                            
                            # Plot trend
                            trend_fig = px.line(
                                forecast_comp, 
                                x='ds', 
                                y='trend',
                                title="Trend Component",
                                labels={'ds': 'Date', 'trend': 'Trend'},
                                template='plotly_white'
                            )
                            st.plotly_chart(trend_fig, use_container_width=True)
                            
                            # Plot yearly seasonality if present
                            if 'yearly' in forecast_comp.columns:
                                yearly_fig = px.line(
                                    forecast_comp, 
                                    x=[d.dayofyear for d in forecast_comp['ds']], 
                                    y='yearly',
                                    title="Yearly Seasonality",
                                    labels={'x': 'Day of Year', 'yearly': 'Effect'},
                                    template='plotly_white'
                                )
                                st.plotly_chart(yearly_fig, use_container_width=True)
                            
                            # Plot weekly seasonality if present
                            if 'weekly' in forecast_comp.columns:
                                weekly_fig = px.line(
                                    forecast_comp, 
                                    x=[d.dayofweek for d in forecast_comp['ds']], 
                                    y='weekly',
                                    title="Weekly Seasonality",
                                    labels={'x': 'Day of Week (0=Monday)', 'weekly': 'Effect'},
                                    template='plotly_white'
                                )
                                st.plotly_chart(weekly_fig, use_container_width=True)
                                
                        except Exception as comp_error:
                            st.error(f"Error plotting components: {str(comp_error)}")
                    
                    # Display forecast metrics
                    st.subheader("Forecast Metrics")
                    
                    # Calculate mean absolute percentage error (MAPE) for historical data
                    historical_dates = forecast_df['Date'].isin(forecast['ds'])
                    actual = forecast_df.loc[historical_dates, 'Price'].values
                    predicted = forecast.loc[forecast['ds'].isin(forecast_df['Date']), 'yhat'].values
                    
                    if len(actual) > 0 and len(predicted) > 0:
                        # Calculate errors
                        mape = 100 * np.mean(np.abs((actual - predicted) / actual))
                        mae = np.mean(np.abs(actual - predicted))
                        
                        # Display metrics
                        metrics_col1, metrics_col2 = st.columns(2)
                        metrics_col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
                        metrics_col2.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
                    
                    # Display future predictions table
                    st.subheader("Forecast Values")
                    future_forecast = forecast[forecast['ds'] > forecast_df['Date'].max()]
                    future_forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                    future_forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                    future_forecast_display = future_forecast_display.round(2)
                    st.dataframe(future_forecast_display, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    st.text(traceback.format_exc())
                    st.info("Forecasting works best with daily data over longer periods. Try using a different timeframe or commodity.")
            
    with tab4:
        st.header(f"{selected_commodity.capitalize()} Data Explorer")
        
        # Show the data table
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export options
        export_format = st.selectbox(
            "Export Format", 
            options=["CSV", "Excel"]
        )
        
        if st.button("Export Data"):
            if export_format == "CSV":
                csv = filtered_df.to_csv(index=False)
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                filename = f"{selected_commodity}_{current_date}.csv"
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:  # Excel
                # Need to use BytesIO for Excel files
                import io
                buffer = io.BytesIO()
                filtered_df.to_excel(buffer, index=False)
                buffer.seek(0)
                
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                filename = f"{selected_commodity}_{current_date}.xlsx"
                
                st.download_button(
                    label="Download Excel",
                    data=buffer,
                    file_name=filename,
                    mime="application/vnd.ms-excel"
                )
else:
    st.error(f"Selected commodity '{selected_commodity}' not found in data.")
