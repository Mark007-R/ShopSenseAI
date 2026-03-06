# UI Quick Start Guide

## Launch the UI

**Option 1 - Double-click (Windows):**
```
Double-click: ui/run_ui.bat
```

**Option 2 - Command Line:**
```bash
cd ui
python recommendation_ui.py
```

**Option 3 - From main folder:**
```bash
python ui/recommendation_ui.py
```

## UI Components

### 1. System Status Bar
- Shows "System Ready" when loaded (green)
- Shows "Loading..." during initialization (orange)
- Shows "Error" if something fails (red)

### 2. Input Section

**For Existing Users:**
- User ID: Enter any user ID (default one is pre-filled)
- Method: Choose from dropdown:
  - `user_cf` - User-based Collaborative Filtering
  - `item_cf` - Item-based Collaborative Filtering  
  - `content` - Content-based Filtering
  - `hybrid` - Hybrid (combines all methods)
- Number of Recommendations: Use spinner (1-50)

**For New Users:**
- Items: Enter product names, categories, or brands
  - Example: `smart watch, headphones, electronics`
  - Separate with commas

### 3. Action Buttons
- **Generate Recommendations**: Click to get recommendations
- **Clear**: Clear the results area
- **Save to CSV**: Export results to a CSV file

### 4. Results Area
- Shows recommendations in a formatted table
- Displays: Rank, Product ID, Product Name, Category, Score
- Scrollable text area for easy viewing

## Example Workflow

### Existing User Example:
1. Select "Existing User" radio button
2. User ID is pre-filled (or enter your own)
3. Select method: `hybrid`
4. Set recommendations: `10`
5. Click "Generate Recommendations"
6. View results
7. Click "Save to CSV" to export

### New User Example:
1. Select "New User" radio button
2. Enter items: `laptop, gaming, electronics`
3. Set recommendations: `15`
4. Click "Generate Recommendations"
5. View matched items and recommendations
6. Click "Save to CSV" to export

## Tips

- The UI loads the dataset automatically on startup
- Pre-filled values are ready for quick testing
- All recommendation algorithms from the main system work in the UI
- CSV files can be saved anywhere you choose
- The UI is responsive and resizable

## Troubleshooting

**UI doesn't launch:**
- Make sure you're in the correct directory
- Check that Python and tkinter are installed
- Try: `python -m tkinter` to test tkinter

**Error loading system:**
- Ensure the dataset file exists in `datasets/` folder
- Check that all dependencies are installed

**No recommendations shown:**
- Verify the user ID exists in the dataset
- Try a different recommendation method
- Increase the number of recommendations
