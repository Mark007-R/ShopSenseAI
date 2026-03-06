import streamlit as st
import pandas as pd
from recommender import EcommerceRecommender
import os

PRODUCTS_PATH = "data/products.csv"

st.set_page_config(
    page_title="ShopSmart - Your Online Store",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_recommender():
    return EcommerceRecommender(PRODUCTS_PATH)

recommender = load_recommender()

if 'user_id' not in st.session_state:
    st.session_state.user_id = 'user_001'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def format_price(price):
    return f"₹{price:,.0f}"

def display_star_rating(rating):
    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    
    stars = "⭐" * full_stars
    if half_star:
        stars += "⭐"
    stars += "☆" * empty_stars
    return stars

def product_card(product, key_prefix=""):
    # Use actual product image from CSV with Unsplash optimization
    image_url = product["image_url"]
    if "unsplash" in image_url.lower():
        image_url += "&auto=format&fit=crop&w=400&q=80" if "?" in image_url else "?auto=format&fit=crop&w=400&q=80"
    
    discount_badge = ""
    discount_pct = 0
    if product['original_price'] > product['price']:
        discount_pct = int(((product['original_price'] - product['price']) / product['original_price']) * 100)
        discount_badge = f"<div style='position: absolute; top: 10px; right: 10px; background: #1DB954; color: #000; padding: 5px 10px; border-radius: 4px; font-weight: 700; font-size: 12px;'>-{discount_pct}%</div>"
    
    with st.container():
        st.markdown(f"""
        <div style='position: relative; border-radius: 16px; overflow: hidden; background: #1a1a1a; margin-bottom: 20px; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'
             onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 8px 20px rgba(29,185,84,0.3)';"
             onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.3)';">
            {discount_badge}
            <img src='{image_url}' style='width: 100%; height: 220px; object-fit: cover;' onerror="this.src='https://via.placeholder.com/400x220/1a1a1a/1DB954?text={product['brand']}'" />
            <div style='padding: 16px;'>
                <h3 style='margin: 0 0 8px 0; color: #fff; font-size: 15px; font-weight: 600; height: 40px; overflow: hidden; line-height: 1.3;'>{product["name"]}</h3>
                <p style='color: #b3b3b3; font-size: 13px; margin: 0 0 12px 0;'>{product["brand"]}</p>
                <div style='display: flex; align-items: center; margin-bottom: 12px;'>
                    <span style='color: #ffa500; font-size: 14px;'>{'⭐' * int(product['rating'])}</span>
                    <span style='color: #b3b3b3; font-size: 12px; margin-left: 6px;'>({product['reviews_count']})</span>
                </div>
                <div style='margin-bottom: 16px;'>
                    <span style='font-size: 24px; font-weight: 700; color: #1DB954;'>{format_price(product["price"])}</span>
                    {f"<span style='text-decoration: line-through; color: #666; margin-left: 8px; font-size: 14px;'>{format_price(product['original_price'])}</span>" if product['original_price'] > product['price'] else ""}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Details", key=f"view_{key_prefix}_{product['product_id']}", use_container_width=True, type="secondary"):
                st.session_state.selected_product = product['product_id']
                st.session_state.current_page = 'product'
                st.rerun()
        with col2:
            if st.button("Add to Cart", key=f"cart_{key_prefix}_{product['product_id']}", use_container_width=True):
                recommender.add_to_cart(st.session_state.user_id, product['product_id'])
                st.toast("✓ Added to cart!", icon="✅")
                st.rerun()

def product_detail_page():
    product = recommender.get_product_by_id(st.session_state.selected_product)
    
    # Use actual product image with optimizations
    image_url = product['image_url']
    if "unsplash" in image_url.lower():
        image_url += "&auto=format&fit=crop&w=800&q=90" if "?" in image_url else "?auto=format&fit=crop&w=800&q=90"
    
    if st.button("← Back to Shopping", type="secondary"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
            <img src='{image_url}' 
                 style='width: 100%; border-radius: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);' 
                 onerror="this.src='https://via.placeholder.com/800x600/1a1a1a/1DB954?text={product['brand']}'" />
        """, unsafe_allow_html=True)
    
    with col2:
        st.title(product['name'])
        st.subheader(f"by {product['brand']}")
        
        st.markdown(f"### {format_price(product['price'])}")
        if product['original_price'] > product['price']:
            discount_pct = int(((product['original_price'] - product['price']) / product['original_price']) * 100)
            st.markdown(f"<span style='text-decoration: line-through; color: #B3B3B3;'>{format_price(product['original_price'])}</span> <span style='color: #1DB954; font-weight: bold; font-size: 18px;'>-{discount_pct}% OFF</span>", unsafe_allow_html=True)
        
        st.markdown(f"### {display_star_rating(product['rating'])} {product['rating']}/5")
        st.caption(f"{product['reviews_count']} ratings")
        
        st.markdown("---")
        
        if product['stock'] > 0:
            st.success(f"✓ In Stock ({product['stock']} available)")
            
            quantity = st.number_input("Quantity", min_value=1, max_value=product['stock'], value=1)
            
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🛒 Add to Cart", use_container_width=True, type="primary"):
                    recommender.add_to_cart(st.session_state.user_id, product['product_id'], quantity)
                    st.success(f"✓ Added {quantity} item(s) to cart!")
                    st.rerun()
            with col_btn2:
                st.markdown("""
                    <style>
                    .wishlist-btn button {
                        background: transparent !important;
                        border: 2px solid #1DB954 !important;
                        color: #1DB954 !important;
                        font-weight: 600 !important;
                    }
                    .wishlist-btn button:hover {
                        background: rgba(29, 185, 84, 0.1) !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                if st.button("❤️ Wishlist", use_container_width=True):
                    recommender.add_to_wishlist(st.session_state.user_id, product['product_id'])
                    st.success("✓ Added to wishlist!")
        else:
            st.error("❌ Out of Stock")
    
    st.markdown("---")
    st.subheader("Product Description")
    st.write(product['description'])
    
    st.markdown("---")
    st.subheader("Key Features")
    features = product['features'].split('|')
    for feature in features:
        st.markdown(f"✓ {feature.strip()}")
    
    st.markdown("---")
    st.subheader("Similar Products")
    similar = recommender.get_similar_products(product['product_id'], n=4)
    if similar:
        cols = st.columns(4)
        for idx, sim_product in enumerate(similar):
            with cols[idx]:
                product_card(sim_product, key_prefix=f"similar_{idx}")
    
    st.markdown("---")
    st.subheader("Frequently Bought Together")
    together = recommender.get_frequently_bought_together(product['product_id'], n=3)
    if together:
        cols = st.columns(3)
        for idx, tog_product in enumerate(together):
            with cols[idx]:
                product_card(tog_product, key_prefix=f"together_{idx}")

def home_page():
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🛍️ ShopSmart Store</h1>", unsafe_allow_html=True)
    
    search_col, filter_col, sort_col = st.columns([3, 1, 1])
    
    with search_col:
        search_query = st.text_input("Search products", placeholder="Search for products, brands, categories...")
    
    with filter_col:
        categories = recommender.get_categories()
        selected_category = st.selectbox("Category", categories)
    
    with sort_col:
        sort_options = {
            'Rating': 'rating',
            'Price: Low to High': 'price_low',
            'Price: High to Low': 'price_high',
            'Most Popular': 'popular'
        }
        sort_by = st.selectbox("Sort By", list(sort_options.keys()))
    
    if search_query:
        products_df = recommender.search_products(search_query)
        st.subheader(f"Search Results for '{search_query}'")
    else:
        products_df = recommender.get_all_products(category=selected_category, sort_by=sort_options[sort_by])
        st.subheader("All Products" if selected_category == 'All' else f"{selected_category}")
    
    if len(products_df) == 0:
        st.info("No products found.")
        return
    
    st.markdown("---")
    st.subheader("Recommended for You")
    recommendations = recommender.get_personalized_recommendations(st.session_state.user_id, n=4)
    if recommendations:
        cols = st.columns(4)
        for idx, rec_product in enumerate(recommendations):
            with cols[idx]:
                product_card(rec_product, key_prefix=f"rec_{idx}")
    
    st.markdown("---")
    st.subheader("Browse Products")
    
    products_per_row = 4
    products_list = products_df.to_dict('records')
    
    for i in range(0, len(products_list), products_per_row):
        cols = st.columns(products_per_row)
        for idx, product in enumerate(products_list[i:i+products_per_row]):
            with cols[idx]:
                product_card(product, key_prefix=f"browse_{i}_{idx}")

def cart_page():
    st.title("🛒 Shopping Cart")
    
    st.markdown("""
        <style>
        .continue-btn button {
            background: transparent !important;
            border: 2px solid #1DB954 !important;
            color: #1DB954 !important;
        }
        .continue-btn button:hover {
            background: #1DB954 !important;
            color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("← Continue Shopping"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    cart_items = recommender.get_cart(st.session_state.user_id)
    
    if not cart_items:
        st.info("Your cart is empty. Start shopping!")
        return
    
    for item in cart_items:
        image_url = item['image_url'] + "&auto=format&fit=crop&w=200&q=80"
        
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            st.markdown(f"""
                <img src='{image_url}' 
                     style='width: 100px; height: 100px; object-fit: cover; border-radius: 8px;' 
                     onerror=\"this.src='https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=200&auto=format&fit=crop&q=80'\"/>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{item['name']}**")
            st.caption(f"{item['brand']}")
            st.caption(f"{display_star_rating(item['rating'])} ({item['reviews_count']})")
        
        with col3:
            st.markdown(f"**{format_price(item['price'])}**")
            st.caption(f"Qty: {item['quantity']}")
        
        with col4:
            st.markdown(f"**{format_price(item['subtotal'])}**")
            st.markdown("""
                <style>
                .remove-btn button {
                    background: transparent !important;
                    border: 1px solid #ff4444 !important;
                    color: #ff4444 !important;
                    font-size: 12px !important;
                    padding: 4px 12px !important;
                }
                .remove-btn button:hover {
                    background: #ff4444 !important;
                    color: #000000 !important;
                }
                </style>
            """, unsafe_allow_html=True)
            if st.button("Remove", key=f"remove_{item['product_id']}"):
                recommender.remove_from_cart(st.session_state.user_id, item['product_id'])
                st.rerun()
        
        st.markdown("---")
    
    total = recommender.get_cart_total(st.session_state.user_id)
    
    st.markdown("### Order Summary")
    st.markdown(f"<h3 style='color: #1DB954;'>Subtotal: {format_price(total)}</h3>", unsafe_allow_html=True)
    st.markdown(f"**Shipping:** <span style='color: #1DB954;'>FREE</span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: #1DB954;'>Total: {format_price(total)}</h2>", unsafe_allow_html=True)
    
    if st.button("✓ Proceed to Checkout", type="primary", use_container_width=True):
        st.balloons()
        st.success("🎉 Order placed successfully! Thank you for shopping with us!")

def wishlist_page():
    st.title("❤️ My Wishlist")
    
    if st.button("← Back to Shopping"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    wishlist_items = recommender.get_wishlist(st.session_state.user_id)
    
    if not wishlist_items:
        st.info("Your wishlist is empty.")
        return
    
    products_per_row = 4
    for i in range(0, len(wishlist_items), products_per_row):
        cols = st.columns(products_per_row)
        for idx, product in enumerate(wishlist_items[i:i+products_per_row]):
            with cols[idx]:
                product_card(product, key_prefix=f"wishlist_{i}_{idx}")
                st.markdown("""
                    <style>
                    .wishlist-remove-btn button {
                        background: transparent !important;
                        border: 1px solid #ff4444 !important;
                        color: #ff4444 !important;
                        padding: 6px 12px !important;
                        font-size: 12px !important;
                    }
                    .wishlist-remove-btn button:hover {
                        background: #ff4444 !important;
                        color: #000000 !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                if st.button("🗑️ Remove", key=f"remove_wish_{product['product_id']}", use_container_width=True):
                    recommender.remove_from_wishlist(st.session_state.user_id, product['product_id'])
                    st.rerun()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        padding: 2rem;
    }
    
    h1 {color: #FFFFFF; font-weight: 700; margin-bottom: 1.5rem;}
    h2 {color: #FFFFFF; font-weight: 600; margin-bottom: 1rem;}
    h3 {color: #FFFFFF; font-weight: 600;}
    h4 {color: #B3B3B3; font-weight: 500;}
    p, label, .stMarkdown {color: #B3B3B3;}
    
    /* Primary Buttons (Add to Cart, Checkout) */
    .stButton>button[kind="primary"],
    .stButton>button:not([kind="secondary"]) {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(29, 185, 84, 0.3) !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }
    .stButton>button[kind="primary"]:hover,
    .stButton>button:not([kind="secondary"]):hover {
        background: linear-gradient(90deg, #1ed760 0%, #1fdf64 100%) !important;
        box-shadow: 0 4px 12px rgba(29, 185, 84, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary Buttons (View Details) */
    .stButton>button[kind="secondary"] {
        background: transparent !important;
        color: #FFFFFF !important;
        border: 1.5px solid #333 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button[kind="secondary"]:hover {
        background: #222 !important;
        border-color: #1DB954 !important;
        color: #1DB954 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stSelectbox>div>div {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
        border: 1.5px solid #333 !important;
        border-radius: 8px !important;
        padding: 0.6rem !important;
    }
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div:focus {
        border-color: #1DB954 !important;
        box-shadow: 0 0 0 2px rgba(29, 185, 84, 0.2) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #1a1a1a !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    /* Success Messages */
    .stSuccess {
        background-color: rgba(29, 185, 84, 0.15) !important;
        color: #1DB954 !important;
        border: 1px solid #1DB954 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Number Input */
    .stNumberInput>div>div>input {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
        border: 1.5px solid #333 !important;
        border-radius: 8px !important;
    }
    
    /* Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>🛍️ ShopSmart</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    cart_count = recommender.get_cart_count(st.session_state.user_id)
    
    if st.button(f"🛒 Cart ({cart_count})", use_container_width=True, type="primary"):
        st.session_state.current_page = 'cart'
        st.rerun()
    
    if st.button("❤️ Wishlist", use_container_width=True):
        st.session_state.current_page = 'wishlist'
        st.rerun()
    
    if st.button("🏠 Home", use_container_width=True, type="secondary"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("---")
    st.markdown("<h4 style='margin-bottom: 15px;'>Shop by Category</h4>", unsafe_allow_html=True)
    categories = recommender.get_categories()
    for category in categories:
        if category != 'All':
            st.markdown(f"<p style='color: #b3b3b3; margin: 8px 0;'>• {category}</p>", unsafe_allow_html=True)

if st.session_state.current_page == 'home':
    home_page()
elif st.session_state.current_page == 'product':
    product_detail_page()
elif st.session_state.current_page == 'cart':
    cart_page()
elif st.session_state.current_page == 'wishlist':
    wishlist_page()
