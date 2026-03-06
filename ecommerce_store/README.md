# ShopSmart E-Commerce Store

An Amazon-like e-commerce website with AI-powered product recommendations, built with Streamlit.

## Features

### Shopping Experience
- **Product Catalog**: Browse 20+ products across Electronics, Appliances, and Smart Home categories
- **Product Details**: View detailed information, images, ratings, and reviews
- **Search & Filter**: Search products and filter by category
- **Sort Options**: Sort by rating, price, or popularity

### Recommendations
- **Personalized Recommendations**: AI-powered suggestions based on browsing history
- **Similar Products**: Find products similar to what you're viewing
- **Frequently Bought Together**: See what other customers purchase together

### Shopping Features
- **Shopping Cart**: Add products, adjust quantities, and view total
- **Wishlist**: Save products for later
- **Product Images**: High-quality images from Unsplash
- **Ratings & Reviews**: See what others think before buying

## Quick Start

1. **Navigate to the ecommerce_store folder**:
```bash
cd ecommerce_store
```

2. **Run the app**:
```bash
streamlit run app.py
```

3. **Open your browser** to the URL shown (usually http://localhost:8501)

## File Structure

```
ecommerce_store/
├── app.py                  # Main Streamlit application
├── recommender.py          # Recommendation engine
├── create_products.py      # Product catalog generator
├── data/
│   └── products.csv        # Product catalog (20 products)
├── static/
│   └── images/             # Product images directory
└── README.md               # This file
```

## How It Works

### Recommendation Engine
The system uses multiple recommendation strategies:

1. **Content-Based Filtering**: 
   - Uses TF-IDF on product names, categories, brands, and features
   - Finds similar products based on text similarity

2. **Collaborative Filtering**:
   - Tracks user browsing and purchase history
   - Recommends products based on similar user patterns

3. **Hybrid Approach**:
   - Combines content similarity with user behavior
   - Falls back to trending products for new users

### Pages

**Home Page**:
- Personalized recommendations at the top
- Full product catalog with filtering and sorting
- Search functionality

**Product Detail Page**:
- Large product image
- Full description and features
- Add to cart/wishlist buttons
- Similar products section
- Frequently bought together section

**Shopping Cart**:
- View all cart items
- Adjust quantities
- See order total
- Proceed to checkout

**Wishlist**:
- Save products for later
- Move to cart when ready

## Product Catalog

The catalog includes 20 products across three main categories:

### Electronics
- TVs, Smartphones, Laptops, Tablets
- Cameras, Headphones, Speakers

### Appliances
- Refrigerators, Washing Machines, Vacuum Cleaners
- Kitchen appliances (Instant Pot, Air Fryer, Espresso Machine, etc.)

### Smart Home
- Thermostats, Security systems

Each product includes:
- Product ID, Name, Brand
- Category and Subcategory
- Price (current and original)
- Rating and review count
- Stock availability
- High-quality image URL
- Detailed description
- Feature list

## Customization

### Adding More Products

Edit `create_products.py` and add new products to the list, then run:
```bash
python create_products.py
```

### Changing Recommendation Logic

Edit `recommender.py` to adjust:
- Similarity algorithms
- Recommendation weights
- Trending product calculations

### UI Customization

Edit `app.py` to modify:
- Layout and styling
- Number of products per row
- Color scheme
- Page structure

## Technologies Used

- **Streamlit**: Web framework for the UI
- **Pandas**: Data manipulation
- **Scikit-learn**: TF-IDF and cosine similarity
- **NumPy**: Numerical operations

## Future Enhancements

- User authentication and profiles
- Order history tracking
- Product reviews and ratings submission
- Payment integration
- Inventory management
- Real-time stock updates
- Email notifications
- Multi-language support
