import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.cluster.hierarchy import dendrogram, linkage
from math import pi
from datetime import timedelta
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎯 Customer Analytics Dashboard")
st.markdown("### Insights for customer segmentation and market basket analysis")


# Load the data
@st.cache_data
def load_data(): 
    df = pd.read_csv('Sample_superstore_dataset.csv', encoding='latin1')
    df_clean = df.dropna()
    df_clean = df_clean[df_clean['Sales'] > 0]
    df_clean = df_clean.drop_duplicates()
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
    df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'])
    return df_clean

@st.cache_data
def load_data():
    """Loading all the saved analysis data"""
    try:
        # Load customer segments data
        customer_segments = pd.read_csv('customer_segments.csv')
        
        # Load association rules
        association_rules = pd.read_csv('association_rules.csv')
        
        # Load cluster profiles
        cluster_profiles = pd.read_csv('cluster_profiles.csv')
        
        return customer_segments, association_rules, cluster_profiles
    
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.error("Please ensure the following files are in your directory:")
        st.error("- customer_segments.csv")
        st.error("- association_rules.csv") 
        st.error("- cluster_profiles.csv")
        return None, None, None

# Load and clean main transaction data
# This block ensures df_clean is available for RFM calculation


df = pd.read_csv('Sample_superstore_dataset.csv', encoding='latin1')
df_clean = df.dropna()
df_clean = df_clean[df_clean['Sales'] > 0]
df_clean = df_clean.drop_duplicates()
df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'])
df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'])

#Calculate RFM metrics
#Define the reference date
reference_date = df_clean['Order Date'].max() + timedelta(days=1)
print(f"Reference date for RFM calculation: {reference_date}")

# Create RFM table
rfm = df_clean.groupby('Customer ID').agg({
    'Order Date': lambda x: (reference_date - x.max()).days,  # Recency
    'Order ID': 'count',                                      # Frequency
    'Sales': 'sum'                                            # Monetary
})

#rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Reset index
rfm = rfm.reset_index()
# --- Add enhanced RFM features ---
# Average Order Value
rfm['Average_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
# Number of unique products purchased per customer
products_per_customer = df_clean.groupby('Customer ID')['Product Name'].nunique().reset_index()
products_per_customer.columns = ['Customer ID', 'Unique_Products']
rfm = rfm.merge(products_per_customer, on='Customer ID', how='left')
# Total transactions per customer
transactions_per_customer = df_clean.groupby('Customer ID').size().reset_index(name="Total_Transactions")
rfm = rfm.merge(transactions_per_customer, on='Customer ID', how='left')
# Product Diversity
rfm['Product_Diversity'] = rfm['Unique_Products'] / rfm['Total_Transactions']
rfm['Product_Diversity'] = rfm['Product_Diversity'].fillna(0)

   
#create tabs

tab1, tab2, tab3 = st.tabs(["👥Customer Segmentation","🛒 Market Basket Analysis","💼 Business Intelligence Summary"])

# --- Tab 1: Customer Segmentation  ---
with tab1:
    st.markdown("### 📊 Segment Statistics & Business Metrics")
    summary_cols = ['Recency', 'Frequency', 'Monetary', 'Average_Order_Value', 'Unique_Products', 'Product_Diversity']
    st.dataframe(rfm[summary_cols].describe().T, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Order Value", f"${rfm['Average_Order_Value'].mean():.0f}")
        st.metric("Avg Unique Products", f"{rfm['Unique_Products'].mean():.1f}")
    with col2:
        st.metric("Avg Product Diversity", f"{rfm['Product_Diversity'].mean():.2f}")
        st.metric("Median Order Value", f"${rfm['Average_Order_Value'].median():.0f}")
    with col3:
        st.metric("Median Unique Products", f"{rfm['Unique_Products'].median():.1f}")
        st.metric("Median Product Diversity", f"{rfm['Product_Diversity'].median():.2f}")
    st.markdown("---")
    st.markdown("### 🔍 Customer Lookup Tool")
    customer_id = st.text_input("Enter Customer ID to look up:")
    if customer_id:
        match = rfm[rfm['Customer ID'].astype(str) == str(customer_id)]
        if not match.empty:
            st.success(f"Customer ID {customer_id} found!")
            st.write(match.T)
            row = match.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recency", f"{row['Recency']}")
                st.metric("Frequency", f"{row['Frequency']}")
            with col2:
                st.metric("Monetary", f"${row['Monetary']:.2f}")
                st.metric("Avg Order Value", f"${row['Average_Order_Value']:.2f}")
            with col3:
                st.metric("Unique Products", f"{row['Unique_Products']}")
                st.metric("Product Diversity", f"{row['Product_Diversity']:.2f}")
            if 'Cluster' in row:
                st.info(f"Segment/Cluster: {row['Cluster']}")
        else:
            st.error("Customer ID not found.")
    st.markdown("### 🟢 2D PCA Cluster Visualization")
    features = ['Recency', 'Frequency', 'Monetary', 'Average_Order_Value', 'Unique_Products', 'Product_Diversity']
    X = rfm[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    rfm['PC1'] = X_pca[:, 0]
    rfm['PC2'] = X_pca[:, 1]
    color_col = 'Cluster' if 'Cluster' in rfm.columns else 'Monetary'
    fig = px.scatter(
        rfm, x='PC1', y='PC2', color=color_col,
        title='2D PCA Cluster Visualization',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        hover_data=['Customer ID'] if 'Customer ID' in rfm.columns else None
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Market Basket Analysis ---
with tab2:
    st.markdown("### 🛒 Association Rules Explorer")
    # Load association rules and frequent itemsets
    try:
        rules = pd.read_csv('association_rules.csv')
    except Exception as e:
        st.error(f"Could not load association_rules.csv: {e}")
        rules = None
    if rules is not None:
        # Filtering controls
        st.sidebar.header("Filter Rules")
        min_support = st.sidebar.slider("Min Support", float(rules['support'].min()), float(rules['support'].max()), float(rules['support'].min()), 0.01)
        min_confidence = st.sidebar.slider("Min Confidence", float(rules['confidence'].min()), float(rules['confidence'].max()), float(rules['confidence'].min()), 0.01)
        min_lift = st.sidebar.slider("Min Lift", float(rules['lift'].min()), float(rules['lift'].max()), float(rules['lift'].min()), 0.01)
        filtered = rules[(rules['support'] >= min_support) & (rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]
        st.dataframe(filtered[['antecedents','consequents','support','confidence','lift']].sort_values('lift', ascending=False), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🤖 Product Recommendation Engine")
        product_input = st.text_input("Enter a product/category for recommendations:")
        def get_recommendations(product, rules_df, max_recommendations=5):
            relevant_rules = rules_df[rules_df['antecedents'].str.contains(product, case=False, na=False)]
            if len(relevant_rules) == 0:
                return "No recommendations found for this product."
            relevant_rules = relevant_rules.sort_values('lift', ascending=False).head(max_recommendations)
            recommendations = []
            for _, rule in relevant_rules.iterrows():
                consequent = rule['consequents']
                recommendations.append({
                    'Product': consequent,
                    'Confidence': f"{rule['confidence']:.1%}",
                    'Lift': f"{rule['lift']:.2f}"
                })
            return pd.DataFrame(recommendations)
        if product_input:
            st.write(get_recommendations(product_input, filtered))

        st.markdown("---")
        st.markdown("### 📈 Rule Visualization")
        # Parallel coordinates plot for support/confidence/lift
        if not filtered.empty:
            import plotly.express as px
            st.plotly_chart(
                px.parallel_coordinates(
                    filtered,
                    dimensions=['support','confidence','lift'],
                    color='lift',
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    title="Parallel Coordinates: Support, Confidence, Lift"
                ),
                use_container_width=True
            )
        # Network graph visualization (optional, simple version)
        st.markdown("#### Network Graph of Top Rules")
        import networkx as nx
        import matplotlib.pyplot as plt
        top_rules = filtered.sort_values('lift', ascending=False).head(10)
        G = nx.DiGraph()
        for _, row in top_rules.iterrows():
            ant = str(row['antecedents'])
            cons = str(row['consequents'])
            G.add_edge(ant, cons, weight=row['lift'])
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, edge_color='gray', arrows=True)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k,v in labels.items()})
        st.pyplot(plt.gcf())
        plt.clf()

with tab3:
    # ... executive summary, recommendations, ROI projections ...
    st.markdown("## 💼 Executive Summary & Key Insights")
    st.markdown("""
    - The customer base is segmented into distinct clusters based on RFM and product diversity metrics.
    - Top segments include Champions, Loyal Customers, At Risk, and New Customers, each with unique behaviors and value.
    - Market basket analysis reveals strong cross-sell opportunities between key product categories.
    """)
    
    # (Optional) Show cluster/segment summary table
    try:
        cluster_profiles = pd.read_csv('cluster_profiles.csv')
        st.markdown("### 📊 Segment/Cluster Profiles")
        st.dataframe(cluster_profiles, use_container_width=True)
    except Exception as e:
        st.warning("Cluster profiles not found.")

    st.markdown("---")
    st.markdown("## 🎯 Actionable Recommendations for Marketing Teams")
    st.markdown("""
    - **Retain best customers:** Offer VIP programs, exclusive access, and premium service to Champions.
    - **Re-engage at-risk customers:** Send win-back campaigns and personalized offers to high-value but inactive customers.
    - **Develop new/low-engagement customers:** Use onboarding sequences and first-time buyer discounts.
    - **Grow potential loyalists:** Cross-sell complementary products and implement loyalty programs.
    - **Leverage cross-selling:** Use association rules to recommend related products at checkout.
    """)

    st.markdown("---")
    st.markdown("## 💰 ROI Projections for Recommended Strategies")
    # Example ROI calculation using cluster_profiles
    if 'cluster_profiles' in locals() and not cluster_profiles.empty:
        st.markdown("### Estimated Revenue Impact by Segment")
        roi_table = cluster_profiles.copy()
        # Assume 20% response rate for campaign, ROI = size * avg order value * 0.2
        roi_table['Potential_ROI'] = roi_table['Monetary'] / roi_table['Frequency'] * roi_table['Cluster'].map(lambda x: 0.2) * 1  # Simplified
        st.dataframe(roi_table[['Cluster','Potential_ROI']], use_container_width=True)
        st.info("ROI projections are based on average order value and a 20% response rate per segment. Adjust assumptions as needed.")
    else:
        st.warning("ROI projections unavailable: cluster profile data missing.")

    st.markdown("---")
    st.markdown("### 🔗 Top Cross-Selling Opportunities")
    try:
        rules = pd.read_csv('association_rules.csv')
        # Lowered filter thresholds to show more rules
        strong_rules = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.1)]
        top_cross = strong_rules.sort_values('lift', ascending=False).head(10)
        if top_cross.empty:
            st.warning("No strong cross-selling rules found with current thresholds. Showing top 10 rules by lift.")
            top_cross = rules.sort_values('lift', ascending=False).head(10)
        st.dataframe(top_cross[['antecedents','consequents','support','confidence','lift']], use_container_width=True)
    except Exception as e:
        st.warning("Association rules not found or insufficient for cross-sell analysis.")