import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from product_recommendation_system import ProductRecommendationSystem
import config
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Product Recommendation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/product-recommendation',
        'Report a bug': "https://github.com/yourusername/product-recommendation/issues",
        'About': "AI Product Recommendation System - Powered by Machine Learning"
    }
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .recommendation-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_recommendation_system():
    try:
        start_time = time.time()
        logger.info("Loading recommendation system...")
        rec_system = ProductRecommendationSystem(
            data_path=config.DATA_PATH,
            min_purchase_threshold=config.MIN_PURCHASE_THRESHOLD
        )
        load_time = time.time() - start_time
        logger.info(f"System loaded in {load_time:.2f} seconds")
        return rec_system, load_time, None
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        return None, 0, f"Dataset file not found. Please ensure {config.DATA_PATH} exists."
    except Exception as e:
        logger.error(f"Error loading system: {e}")
        return None, 0, f"Error loading system: {str(e)}"

def initialize_session_state():
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'batch_result' not in st.session_state:
        st.session_state.batch_result = None
    if 'request_count' not in st.session_state:
        st.session_state.request_count = 0
    if 'last_request_time' not in st.session_state:
        st.session_state.last_request_time = None

def add_percentage_score(df_recs):
    if len(df_recs) > 0 and 'score' in df_recs.columns:
        max_score = df_recs['score'].max()
        min_score = df_recs['score'].min()
        if max_score > min_score:
            df_recs['confidence_pct'] = ((df_recs['score'] - min_score) / (max_score - min_score) * 100).round(2)
        else:
            df_recs['confidence_pct'] = 100.0
    return df_recs

def main():
    initialize_session_state()
    
    st.title("AI-Powered Product Recommendation Engine")
    st.markdown("### Transform Your E-Commerce Experience with Intelligent Product Suggestions")
    st.markdown("---")
    
    with st.spinner("Initializing recommendation engine..."):
        rec_system, load_time, error = load_recommendation_system()
    
    if error:
        st.error(f"ERROR: {error}")
        st.info("Please check your dataset path in config.py and ensure the file exists.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(rec_system.data):,}")
    with col2:
        st.metric("Users", f"{rec_system.data['user_id'].nunique():,}")
    with col3:
        st.metric("Products", f"{rec_system.data['product_id'].nunique():,}")
    with col4:
        st.metric("Load Time", f"{load_time:.2f}s")
    
    st.markdown("---")
    
    tabs = st.tabs(["Smart Recommendations", "Batch Processing", "Analytics Dashboard", "System Info"])
    
    with tabs[0]:
        get_recommendations_tab(rec_system)
    
    with tabs[1]:
        batch_processing_tab(rec_system)
    
    with tabs[2]:
        analytics_tab(rec_system)
    
    with tabs[3]:
        system_info_tab(rec_system, load_time)

def get_recommendations_tab(rec_system):
    st.header("Get Product Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        
        user_type = st.radio(
            "User Type",
            ["Existing User", "New User"]
        )
        
        method = st.selectbox(
            "Recommendation Method",
            ["hybrid", "user_cf", "item_cf", "content"]
        )
        
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=50,
            value=config.DEFAULT_N_RECOMMENDATIONS
        )
        
        if user_type == "Existing User":
            all_users = rec_system.data['user_id'].unique().tolist()
            user_id = st.selectbox(
                "Select User ID",
                options=all_users
            )
            
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("Generating recommendations..."):
                    start_time = time.time()
                    result = rec_system.get_recommendations(
                        user_id=user_id,
                        method=method,
                        n_recommendations=n_recommendations
                    )
                    processing_time = time.time() - start_time
                    result['processing_time'] = processing_time
                    st.session_state['recommendations'] = result
                    st.session_state['user_type'] = 'existing'
                    st.session_state.request_count += 1
                    st.session_state.last_request_time = datetime.now()
        
        else:
            st.markdown("**Enter items you like (keywords):**")
            seed_items_input = st.text_area(
                "Items (one per line)",
                placeholder="smart watch\nheadphones\nlaptop",
                height=100
            )
            
            if st.button("Get Recommendations", type="primary", use_container_width=True):
                seed_items = [item.strip() for item in seed_items_input.split('\n') if item.strip()]
                
                if not seed_items:
                    st.warning("Please enter at least one item")
                else:
                    with st.spinner("Generating recommendations..."):
                        start_time = time.time()
                        result = rec_system.recommend_for_new_user(
                            seed_items=seed_items,
                            n_recommendations=n_recommendations
                        )
                        processing_time = time.time() - start_time
                        result['processing_time'] = processing_time
                        st.session_state['recommendations'] = result
                        st.session_state['user_type'] = 'new'
                        st.session_state.request_count += 1
                        st.session_state.last_request_time = datetime.now()
    
    with col2:
        st.subheader("Results")
        
        if 'recommendations' in st.session_state and st.session_state['recommendations']:
            result = st.session_state['recommendations']
            
            if st.session_state.get('user_type') == 'new' and 'matched_items' in result:
                st.success(f"Matched items: {', '.join(result['matched_items'])}")
            
            if 'recommendations' in result and len(result['recommendations']) > 0:
                df_recs = pd.DataFrame(result['recommendations'])
                df_recs = add_percentage_score(df_recs)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Recommendations", len(df_recs))
                with col_b:
                    avg_confidence = df_recs['confidence_pct'].mean() if 'confidence_pct' in df_recs.columns else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                with col_c:
                    processing_time = result.get('processing_time', 0)
                    st.metric("Processing Time", f"{processing_time:.3f}s")
                
                column_config = {
                    "score": st.column_config.NumberColumn(
                        "Score",
                        format="%.4f"
                    )
                }
                
                if 'confidence_pct' in df_recs.columns:
                    column_config["confidence_pct"] = st.column_config.ProgressColumn(
                        "Confidence %",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    )
                
                st.dataframe(
                    df_recs,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                csv = df_recs.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations (CSV)",
                    data=csv,
                    file_name=f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                if len(df_recs) > 0:
                    fig = go.Figure()
                    
                    top_10 = df_recs.head(10)
                    
                    fig.add_trace(go.Bar(
                        x=top_10['product_id'],
                        y=top_10['score'],
                        name='Score',
                        marker_color='#1f77b4',
                        text=top_10['score'].round(4),
                        textposition='auto',
                    ))
                    
                    if 'confidence_pct' in top_10.columns:
                        fig.add_trace(go.Scatter(
                            x=top_10['product_id'],
                            y=top_10['confidence_pct'],
                            name='Confidence %',
                            yaxis='y2',
                            marker_color='#2ca02c',
                            mode='lines+markers',
                            line=dict(width=3)
                        ))
                    
                    fig.update_layout(
                        title='Top 10 Recommendations',
                        xaxis_title='Product ID',
                        yaxis_title='Score',
                        yaxis2=dict(
                            title='Confidence %',
                            overlaying='y',
                            side='right'
                        ),
                        hovermode='x unified',
                        xaxis_tickangle=-45,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommendations generated yet. Configure settings and click 'Get Recommendations'")
        else:
            st.info("No recommendations generated yet. Configure settings and click 'Get Recommendations'")

def batch_processing_tab(rec_system):
    st.header("Batch Processing")
    st.markdown("Generate recommendations for multiple users at once")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        
        method = st.selectbox(
            "Recommendation Method",
            ["hybrid", "user_cf", "item_cf", "content"],
            key="batch_method"
        )
        
        n_recommendations = st.slider(
            "Recommendations per User",
            min_value=1,
            max_value=50,
            value=10,
            key="batch_n_recs"
        )
        
        user_limit = st.number_input(
            "User Limit",
            min_value=1,
            max_value=len(rec_system.data['user_id'].unique()),
            value=min(100, len(rec_system.data['user_id'].unique()))
        )
        
        output_filename = st.text_input(
            "Output Filename",
            value=f"batch_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if st.button("Generate Batch Recommendations", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing batch recommendations...")
            progress_bar.progress(30)
            
            try:
                start_time = time.time()
                selected_users = rec_system.data['user_id'].unique()[:int(user_limit)].tolist()
                
                result = rec_system.generate_batch_recommendations(
                    user_ids=selected_users,
                    method=method,
                    n_recommendations=n_recommendations,
                    output_path=output_filename
                )
                
                processing_time = time.time() - start_time
                
                result = add_percentage_score(result)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['batch_result'] = result
                st.session_state['batch_processing_time'] = processing_time
                st.success(f"Generated {len(result)} recommendations for {result['user_id'].nunique()} users in {processing_time:.2f}s")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during batch processing: {str(e)}")
                logger.error(f"Batch processing error: {e}", exc_info=True)
    
    with col2:
        st.subheader("Results")
        
        if 'batch_result' in st.session_state and st.session_state['batch_result'] is not None:
            df_batch = st.session_state['batch_result']
            processing_time = st.session_state.get('batch_processing_time', 0)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Total Recommendations", f"{len(df_batch):,}")
            with col_b:
                st.metric("Unique Users", f"{df_batch['user_id'].nunique():,}")
            with col_c:
                st.metric("Unique Products", f"{df_batch['product_id'].nunique():,}")
            with col_d:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            column_config = {
                "score": st.column_config.NumberColumn(
                    "Score",
                    format="%.4f"
                )
            }
            
            if 'confidence_pct' in df_batch.columns:
                column_config["confidence_pct"] = st.column_config.ProgressColumn(
                    "Confidence %",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100
                )
            
            st.dataframe(
                df_batch, 
                use_container_width=True, 
                hide_index=True, 
                height=400,
                column_config=column_config
            )
            
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Batch Results (CSV)",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            user_counts = df_batch['user_id'].value_counts().head(10)
            fig = px.bar(
                x=user_counts.index,
                y=user_counts.values,
                title='Top 10 Users by Recommendation Count',
                labels={'x': 'User ID', 'y': 'Count'},
                color=user_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if 'confidence_pct' in df_batch.columns:
                fig2 = px.histogram(
                    df_batch,
                    x='confidence_pct',
                    nbins=20,
                    title='Confidence Score Distribution',
                    labels={'confidence_pct': 'Confidence %', 'count': 'Frequency'},
                    color_discrete_sequence=['#2ca02c']
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No batch results yet. Configure settings and generate recommendations.")

def analytics_tab(rec_system):
    st.header("Dataset Analytics")
    
    df = rec_system.data
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Interactions", f"{len(df):,}")
    with col2:
        st.metric("Unique Users", f"{df['user_id'].nunique():,}")
    with col3:
        st.metric("Unique Products", f"{df['product_id'].nunique():,}")
    with col4:
        st.metric("Categories", f"{df['category'].nunique():,}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Interaction Types")
        interaction_counts = df['interaction_type'].value_counts()
        fig1 = px.pie(
            values=interaction_counts.values,
            names=interaction_counts.index,
            title='Distribution of Interaction Types',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Top 10 Categories")
        category_counts = df['category'].value_counts().head(10)
        fig3 = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title='Top 10 Product Categories',
            labels={'x': 'Count', 'y': 'Category'},
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_right:
        st.subheader("User Segments")
        segment_counts = df['user_segment'].value_counts()
        fig2 = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            title='User Segment Distribution',
            labels={'x': 'Segment', 'y': 'Count'},
            color=segment_counts.values,
            color_continuous_scale='Blues'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Top 10 Brands")
        brand_counts = df['brand'].value_counts().head(10)
        fig4 = px.bar(
            x=brand_counts.index,
            y=brand_counts.values,
            title='Top 10 Brands',
            labels={'x': 'Brand', 'y': 'Count'},
            color=brand_counts.values,
            color_continuous_scale='Oranges'
        )
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Raw Data Sample")
    
    show_columns = st.multiselect(
        "Select columns to display",
        options=df.columns.tolist(),
        default=['user_id', 'product_id', 'product_name', 'category', 'brand', 'interaction_type', 'rating']
    )
    
    if show_columns:
        num_rows = st.slider("Number of rows to display", 10, 1000, 100)
        st.dataframe(df[show_columns].head(num_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one column to display")

def system_info_tab(rec_system, load_time):
    st.header("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Metrics")
        st.metric("System Load Time", f"{load_time:.3f} seconds")
        st.metric("Total Requests", st.session_state.request_count)
        
        if st.session_state.last_request_time:
            st.metric("Last Request", st.session_state.last_request_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        st.subheader("Configuration")
        st.write(f"**Data Path:** {config.DATA_PATH}")
        st.write(f"**Default Method:** {config.DEFAULT_METHOD}")
        st.write(f"**Default Recommendations:** {config.DEFAULT_N_RECOMMENDATIONS}")
        st.write(f"**Min Purchase Threshold:** {config.MIN_PURCHASE_THRESHOLD}")
        
        if hasattr(config, 'HYBRID_WEIGHTS'):
            st.write("**Hybrid Weights:**")
            for key, value in config.HYBRID_WEIGHTS.items():
                st.write(f"  - {key}: {value}")
    
    with col2:
        st.subheader("Dataset Information")
        df_info = {
            "Metric": [
                "Total Records",
                "Unique Users",
                "Unique Products",
                "Unique Categories",
                "Unique Brands",
                "Date Range",
                "Memory Usage (MB)"
            ],
            "Value": [
                f"{len(rec_system.data):,}",
                f"{rec_system.data['user_id'].nunique():,}",
                f"{rec_system.data['product_id'].nunique():,}",
                f"{rec_system.data['category'].nunique():,}",
                f"{rec_system.data['brand'].nunique():,}",
                f"{rec_system.data['interaction_date'].min()} to {rec_system.data['interaction_date'].max()}",
                f"{rec_system.data.memory_usage(deep=True).sum() / 1024**2:.2f}"
            ]
        }
        st.table(pd.DataFrame(df_info))
        
        st.subheader("Algorithm Methods")
        methods_info = {
            "Method": ["hybrid", "user_cf", "item_cf", "content"],
            "Description": [
                "Combines all methods with weighted scores",
                "User-based collaborative filtering",
                "Item-based collaborative filtering",
                "Content-based filtering using TF-IDF"
            ]
        }
        st.table(pd.DataFrame(methods_info))

if __name__ == "__main__":
    main()
