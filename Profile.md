# commodity-price-tracker

This project is a web application that tracks and visualizes metal and oil commodity prices over time. It retrieves price data for Copper, Aluminium, Steel, and Oil from Investing.com and provides interactive charts and analysis tools.

## Name: Ian Joscelyne 
## GitHub: BossJossUK 

## Features

- Automated data collection from Investing.com API
- Interactive price charts with multiple visualization types (line, candlestick, area)
- Multi-timeframe analysis (daily, weekly, monthly)
- Commodity price comparison tools
- Normalized price comparison
- Data export in CSV and Excel formats
- Price statistics and analytics

## Requirements

This project uses the following dependencies:

- streamlit==1.31.0
- pandas==2.0.3
- requests==2.31.0
- matplotlib==3.7.2
- schedule==1.2.0
- openpyxl==3.1.2
- plotly==5.18.0
- python-dateutil>=2.8.2

## How to Run

To run the app locally, ensure that you have Python installed and use the following commands:

```bash
# Clone the repository
git clone https://github.com/BossJossUK/commodity-price-tracker.git

# Navigate to the project directory
cd commodity-price-tracker

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
