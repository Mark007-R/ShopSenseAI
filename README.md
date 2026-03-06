# Product Recommendation System

## Overview

A production-grade hybrid recommendation system combining collaborative filtering and content-based filtering for e-commerce product recommendations. Features an interactive **Streamlit web interface** with real-time analytics and percentage-based confidence scores.

## Key Features

- **Interactive Streamlit Web UI** with professional design
- **Confidence Percentage Scores** for all recommendations
- **Multiple Recommendation Algorithms**:
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
  - Content-Based Filtering (TF-IDF)
  - Hybrid Approach with configurable weights
- **Cold-Start Handling** for new users
- **Batch Processing** with progress tracking
- **Real-time Analytics Dashboard**
- **Interactive Visualizations** using Plotly
- **Performance Monitoring** and caching
- **Error Handling** and logging
- **Export Capabilities** (CSV downloads)

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd Hackathon-Cloudfronts
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn scipy streamlit plotly
```

## Usage

### Web Application (Recommended)

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

Or use the batch file:
```bash
run_app.bat
```

The application will open in your browser at `http://localhost:8501`

### Application Features

#### 1. Smart Recommendations Tab
- Get personalized recommendations for existing users
- Generate recommendations for new users based on preferences
- View confidence percentage scores
- Interactive dual-axis charts showing scores and confidence
- Real-time processing time metrics
- Export results to CSV

#### 2. Batch Processing Tab
- Process multiple users simultaneously
- Configurable recommendation counts
- Progress tracking
- Confidence score distribution analysis
- Batch export functionality

#### 3. Analytics Dashboard
- Dataset statistics and insights
- Interactive charts:
  - Interaction type distribution
  - User segment analysis
  - Top categories and brands
- Raw data browser with column selection

#### 4. System Info Tab
- System performance metrics
- Configuration details
- Dataset information
- Algorithm method descriptions

### Command Line Interface

For quick CLI access:

```bash
python QUICKSTART.py
```

### Python API

```python
from product_recommendation_system import ProductRecommendationSystem
import config

rec_system = ProductRecommendationSystem(config.DATA_PATH)

result = rec_system.get_recommendations(
    user_id='U00001', 
    method='hybrid', 
    n_recommendations=10
)

for rec in result['recommendations']:
    print(f"{rec['product_name']}: {rec['score']:.4f} (Confidence: {rec.get('normalized_score', 0):.2f}%)")
```

## Configuration

Edit `config.py` to customize system behavior:

```python
DATA_PATH = "datasets/product_recommendation_dataset_v2.csv"
DEFAULT_METHOD = "hybrid"
DEFAULT_N_RECOMMENDATIONS = 20
DEFAULT_N_SIMILAR_USERS = 10
HYBRID_WEIGHTS = {
    "collaborative_user": 0.3,
    "collaborative_item": 0.3,
    "content_based": 0.4,
}
MIN_PURCHASE_THRESHOLD = 1
```

## Project Structure

```
Hackathon-Cloudfronts/
├── app.py                              Main Streamlit application
├── product_recommendation_system.py    Core recommendation engine
├── config.py                           Configuration settings
├── QUICKSTART.py                       CLI interface
├── requirements.txt                    Python dependencies
├── README.md                           Documentation
├── run_app.bat                         Windows launcher
├── datasets/                           Dataset folder
│   └── product_recommendation_dataset_v2.csv
└── backup/                             Backup files
```

## Recommendation Algorithms

### User-Based Collaborative Filtering
- Identifies similar users based on interaction patterns
- Uses cosine similarity for user comparison
- Recommends products preferred by similar users

### Item-Based Collaborative Filtering
- Finds similar products based on user interactions
- Uses cosine similarity for product comparison
- Recommends items similar to user's previous choices

### Content-Based Filtering
- Analyzes product attributes (category, brand, name)
- Uses TF-IDF vectorization for text features
- Recommends products with similar content characteristics

### Hybrid Approach
- Combines all three methods with configurable weights
- Normalizes scores across different algorithms
- Provides balanced recommendations leveraging multiple signals

## Performance Features

- **Caching**: System initialization is cached for faster subsequent runs
- **Lazy Loading**: Data is loaded only when needed
- **Batch Processing**: Efficient handling of multiple recommendations
- **Progress Tracking**: Real-time updates during batch operations
- **Memory Optimization**: Efficient pandas operations

## Production Features

- **Logging**: Comprehensive logging for debugging and monitoring
- **Error Handling**: Graceful error handling with user-friendly messages
- **Session State**: Maintains user state across interactions
- **Metrics Tracking**: Request counting and timing
- **Export Functions**: CSV download for all results
- **Responsive Design**: Professional UI with custom CSS
- **Confidence Scores**: Percentage-based confidence for all recommendations

## Technical Stack

- **Backend**: Python 3.11+
- **ML Libraries**: scikit-learn, pandas, numpy, scipy
- **Web Framework**: Streamlit
- **Visualization**: Plotly
- **Algorithms**: Cosine Similarity, TF-IDF, Matrix Factorization

## Dataset Requirements

The system expects a CSV file with the following columns:
- `user_id`: Unique user identifier
- `product_id`: Unique product identifier
- `product_name`: Product name
- `category`: Product category
- `brand`: Product brand
- `interaction_type`: Type of interaction (purchase, view, wishlist, add_to_cart)
- `rating`: User rating (optional)
- `interaction_score`: Calculated interaction strength
- Additional metadata fields

## License

Open source

## Support

For issues or questions, please check the documentation or raise an issue in the repository.

## License

Open source
