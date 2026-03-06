# Product Recommendation System - UI

Simple graphical user interface for the Product Recommendation System using Tkinter.

## How to Run

```bash
cd ui
python recommendation_ui.py
```

Or from the main folder:
```bash
python ui/recommendation_ui.py
```

## Features

- **User Type Selection**: Choose between existing user or new user
- **Existing User Mode**: 
  - Enter user ID
  - Select recommendation method (user_cf, item_cf, content, hybrid)
  - Adjust number of recommendations
- **New User Mode**:
  - Enter product preferences (comma-separated)
  - System finds similar products
- **Results Display**: 
  - Shows recommendations in a formatted table
  - Displays product ID, name, category, and score
- **Save to CSV**: Export recommendations to a CSV file

## Requirements

- tkinter (usually comes with Python)
- All dependencies from main project (pandas, numpy, scikit-learn)

## Screenshot Layout

```
┌─────────────────────────────────────────┐
│  Product Recommendation System          │
├─────────────────────────────────────────┤
│  System Status: Ready                   │
├─────────────────────────────────────────┤
│  Input                                  │
│  ○ Existing User  ○ New User            │
│  User ID: [____________]                │
│  Method: [hybrid ▼]                     │
│  Number of Recs: [20]                   │
├─────────────────────────────────────────┤
│  [Generate] [Clear] [Save to CSV]       │
├─────────────────────────────────────────┤
│  Recommendations                        │
│  ┌───────────────────────────────────┐  │
│  │ Rank | Product | Name | Score    │  │
│  │ 1    | P001    | ...  | 0.9234  │  │
│  │ 2    | P045    | ...  | 0.8901  │  │
│  │ ...                               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Usage Steps

1. Launch the application
2. Wait for "System Ready" status
3. Select user type (existing/new)
4. Fill in user ID or item preferences
5. Choose method and number of recommendations
6. Click "Generate Recommendations"
7. View results in the text area
8. Optionally save to CSV

## Notes

- The UI automatically loads the dataset from the parent directory
- Default values are pre-filled for quick testing
- All recommendation methods from the main system are available
