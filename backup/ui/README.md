# Product Recommendation System - Modern UI

Professional and feature-rich graphical interface for the Product Recommendation System.

## 🚀 How to Run

**Option 1 - Double-click (Windows):**
```
Double-click: run_ui.bat
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

## ✨ Features

### 🎁 Get Recommendations Tab
- **Modern UI Design**: Clean, professional interface with color-coded sections
- **User Type Selection**: Easy toggle between existing and new users
- **Multiple Algorithms**: 
  - 🤝 User-Based Collaborative Filtering
  - 📦 Item-Based Collaborative Filtering
  - 🎨 Content-Based Filtering
  - ⚡ Hybrid (Best - combines all methods)
- **Interactive Results Table**: 
  - Sortable columns
  - Color-coded rows for better readability
  - Shows: Rank, Product ID, Name, Category, Brand, Price, Rating, Score
- **Real-time Status**: Live system status and statistics display
- **One-Click Export**: Save recommendations to CSV with file dialog

### 📊 Batch Processing Tab
- **Bulk Recommendations**: Generate recommendations for multiple users at once
- **Customizable Parameters**:
  - Number of users (1-1000)
  - Recommendations per user (1-50)
  - Algorithm selection
- **Progress Indicator**: Visual feedback during processing
- **Automatic Export**: Save batch results to CSV

### 📈 Analytics Tab
- **Comprehensive Statistics**:
  - Dataset overview (records, users, products)
  - Interaction type breakdown
  - Top 10 categories and brands
  - Rating statistics
  - Price statistics
  - User segment analysis
- **Formatted Dashboard**: Easy-to-read analytics display

## 🎨 Design Highlights

- **Modern Color Scheme**: Professional blue/gray palette
- **Responsive Layout**: Resizable window with proper scaling
- **Icon Integration**: Emoji icons for better visual navigation
- **Tab-based Navigation**: Organized workflow across features
- **Threading Support**: Non-blocking operations for smooth UX
- **Error Handling**: User-friendly error messages

## 📋 Requirements

- Python 3.8+
- tkinter (included with Python)
- pandas, numpy, scikit-learn (from main project)

## 💡 Usage Examples

### Get Single User Recommendations:
1. Go to "🎁 Get Recommendations" tab
2. Select "👤 Existing User"
3. Enter or use pre-filled User ID
4. Choose algorithm (Hybrid recommended)
5. Set number of recommendations (default: 20)
6. Click "🚀 Generate"
7. View results in interactive table
8. Click "💾 Save CSV" to export

### New User Recommendations:
1. Select "🆕 New User"
2. Enter items: `laptop, gaming, electronics`
3. Click "🚀 Generate"
4. System finds similar products automatically

### Batch Processing:
1. Go to "📊 Batch Processing" tab
2. Set number of users (e.g., 50)
3. Set recommendations per user (e.g., 10)
4. Choose method (hybrid)
5. Click "🚀 Generate Batch Recommendations"
6. Choose save location in file dialog
7. Get CSV with all recommendations

### View Analytics:
1. Go to "📈 Analytics" tab
2. View comprehensive system statistics
3. Analyze dataset insights

## 🎯 Key Benefits

- **Professional Interface**: Modern, clean design suitable for presentations
- **Fast Performance**: Threaded operations prevent UI freezing
- **Complete Features**: All main system capabilities accessible via UI
- **User-Friendly**: Intuitive navigation with helpful visual cues
- **Production-Ready**: Error handling and validation built-in
