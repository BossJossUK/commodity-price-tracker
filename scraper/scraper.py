import requests
import csv
import json
import pandas as pd
import datetime
import os
import logging
import time
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("commodity_scraper")

# Ensure data directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Base URLs for copper, aluminium, steel, and oil from investing.com
BASE_URL_COPPER = "https://api.investing.com/api/financialdata/historical/8831"
BASE_URL_ALUMINIUM = "https://api.investing.com/api/financialdata/historical/8880"
BASE_URL_STEEL = "https://api.investing.com/api/financialdata/historical/8918"
BASE_URL_OIL = "https://api.investing.com/api/financialdata/historical/8849"

# Headers for API requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.investing.com/",
    "Origin": "https://www.investing.com",
    "Content-Type": "application/json",
    "Domain-Id": "www"
}

# Define mappings between commodity names and their API URLs
COMMODITY_URLS = {
    "copper": BASE_URL_COPPER,
    "aluminium": BASE_URL_ALUMINIUM,
    "steel": BASE_URL_STEEL,
    "oil": BASE_URL_OIL
}

def get_historical_data(base_url, start_date, end_date, time_frame):
    """
    Fetch historical commodity data from investing.com API
    
    Args:
        base_url: API endpoint for the specific commodity
        start_date: Start date for data in YYYY-MM-DD format
        end_date: End date for data in YYYY-MM-DD format
        time_frame: Data resolution (Daily, Weekly, or Monthly)
        
    Returns:
        List of data points or None if request fails
    """
    params = {
        "start-date": start_date,
        "end-date": end_date,
        "time-frame": time_frame,
        "add-missing-rows": "false"
    }
    
    try:
        logger.info(f"Fetching {time_frame} data from {base_url}")
        response = requests.get(base_url, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        return data["data"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None

def write_to_csv(data, filename):
    """Write raw API response data to CSV"""
    file_path = os.path.join(DATA_DIR, filename)
    if data:
        try:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data[0].keys())
                for item in data:
                    writer.writerow(item.values())
            logger.info(f"CSV file '{filename}' created successfully.")
        except IOError as e:
            logger.error(f"IO error: {e}")
    else:
        logger.warning(f"No data available to write to '{filename}'.")

def write_filtered_csv(data, filename, columns, column_mapping):
    """Write formatted and filtered data to CSV"""
    file_path = os.path.join(DATA_DIR, filename)
    if data:
        try:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                for item in reversed(data):  # Reverse to get chronological order
                    filtered_item = {'Date': item['rowDate']}
                    filtered_item.update({column_mapping[key]: item[key] for key in column_mapping if key in item})
                    writer.writerow(filtered_item.values())
            logger.info(f"CSV file '{filename}' created successfully.")
        except KeyError as e:
            logger.error(f"Key error: {e}")
        except IOError as e:
            logger.error(f"IO error: {e}")
    else:
        logger.warning(f"No data available to write to '{filename}'.")

def create_consolidated_weekly_prices():
    """Create a consolidated CSV with weekly prices for all commodities"""
    try:
        # Read the weekly data from CSV files
        aluminium_weekly = pd.read_csv(os.path.join(DATA_DIR, 'aluminium_weekly_data.csv'))
        copper_weekly = pd.read_csv(os.path.join(DATA_DIR, 'copper_weekly_data.csv'))
        steel_weekly = pd.read_csv(os.path.join(DATA_DIR, 'steel_weekly_data.csv'))
        oil_weekly = pd.read_csv(os.path.join(DATA_DIR, 'oil_weekly_data.csv'))

        # Ensure the 'Date' columns are in datetime format for merging
        aluminium_weekly['Date'] = pd.to_datetime(aluminium_weekly['Date'])
        copper_weekly['Date'] = pd.to_datetime(copper_weekly['Date'])
        steel_weekly['Date'] = pd.to_datetime(steel_weekly['Date'])
        oil_weekly['Date'] = pd.to_datetime(oil_weekly['Date'])

        # Merge the dataframes on 'Date' column
        merged_df = pd.merge(aluminium_weekly[['Date', 'Price']], copper_weekly[['Date', 'Price']], on='Date', suffixes=('_Aluminium', '_Copper'))
        merged_df = pd.merge(merged_df, steel_weekly[['Date', 'Price']], on='Date')
        merged_df.rename(columns={'Price': 'Price_Steel'}, inplace=True)
        merged_df = pd.merge(merged_df, oil_weekly[['Date', 'Price']], on='Date')
        merged_df.rename(columns={'Price': 'Price_Oil'}, inplace=True)

        # Rename columns for clarity
        merged_df.rename(columns={
            'Price_Aluminium': 'Price Aluminium', 
            'Price_Copper': 'Price Copper', 
            'Price_Steel': 'Price Steel', 
            'Price_Oil': 'Price Oil'
        }, inplace=True)

        # Write the merged dataframe to a new CSV file
        merged_df.to_csv(os.path.join(DATA_DIR, 'consolidated_weekly_prices.csv'), index=False)
        logger.info("Consolidated weekly prices CSV file created successfully.")
    except Exception as e:
        logger.error(f"Error creating consolidated weekly prices: {e}")

def run_scraper():
    """
    Main function to run the scraper for all commodities
    """
    logger.info("Starting commodity price scraper")
    
    # Columns to be included in the filtered CSV and their mapping
    filtered_columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change %']
    column_mapping = {
        'last_close': 'Price',
        'last_open': 'Open',
        'last_max': 'High',
        'last_min': 'Low',
        'volume': 'Vol',
        'change_precent': 'Change %'
    }
    
    # Get current date
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    # Set start dates for different periods
    start_date_daily = (datetime.datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    start_date_weekly = (datetime.datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    start_date_monthly = (datetime.datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Process each commodity
    for commodity_name, base_url in COMMODITY_URLS.items():
        try:
            # Process daily data
            daily_data = get_historical_data(base_url, start_date_daily, end_date, "Daily")
            write_to_csv(daily_data, f'{commodity_name}_daily_data_raw.csv')
            write_filtered_csv(daily_data, f'{commodity_name}_daily_data.csv', filtered_columns, column_mapping)
            
            # Process weekly data
            weekly_data = get_historical_data(base_url, start_date_weekly, end_date, "Weekly")
            write_to_csv(weekly_data, f'{commodity_name}_weekly_data_raw.csv')
            write_filtered_csv(weekly_data, f'{commodity_name}_weekly_data.csv', filtered_columns, column_mapping)
            
            # Process monthly data
            monthly_data = get_historical_data(base_url, start_date_monthly, end_date, "Monthly")
            write_to_csv(monthly_data, f'{commodity_name}_monthly_data_raw.csv')
            write_filtered_csv(monthly_data, f'{commodity_name}_monthly_data.csv', filtered_columns, column_mapping)
            
            # Add a small delay to avoid hitting API rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {commodity_name}: {e}")
    
    # Create consolidated weekly prices file
    create_consolidated_weekly_prices()
    
    logger.info("Commodity price scraping completed")

if __name__ == "__main__":
    run_scraper()
