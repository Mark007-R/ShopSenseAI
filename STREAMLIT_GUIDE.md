# Streamlit App Quick Start Guide

## Installation Complete! ✓

Your project has been successfully converted to use Streamlit web interface.

## Running the App

### Option 1: Command Line
```bash
streamlit run app.py
```

### Option 2: Batch File (Windows)
Double-click `run_app.bat` or run:
```bash
run_app.bat
```

## Accessing the App

Once started, Streamlit will provide URLs like:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.x.x:8501

Open either URL in your web browser.

## App Features

### 📊 Get Recommendations Tab
- **Existing User**: Select a user ID and get personalized recommendations
- **New User**: Enter product keywords and get recommendations
- Interactive visualizations
- Download results as CSV

### 👥 Batch Processing Tab
- Generate recommendations for multiple users at once
- Configure number of recommendations per user
- Export batch results to CSV
- View distribution charts

### 📈 Analytics Tab
- Dataset statistics and insights
- Interactive charts and graphs
- User segments analysis
- Top categories and brands
- Raw data browser

## Settings

Each tab allows you to customize:
- Recommendation method (hybrid, user_cf, item_cf, content)
- Number of recommendations
- User selection or input

## Tips

1. The system caches the recommendation engine for better performance
2. All results can be downloaded as CSV files
3. Use the sidebar settings to configure recommendations
4. Visualizations update automatically based on your selections

## Troubleshooting

If the app doesn't start:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that port 8501 is not already in use
3. Try running with: `streamlit run app.py --server.port 8502`

## Files

- `app.py` - Main Streamlit application
- `product_recommendation_system.py` - Core recommendation engine
- `config.py` - Configuration settings
- `requirements.txt` - Python dependencies

Enjoy your new web-based recommendation system!
