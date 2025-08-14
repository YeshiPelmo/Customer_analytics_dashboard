# Customer_analytics_dashboard
Please find the deployed app here: https://customer-insights-platform.streamlit.app/
This interactive customer Analytics Dashboard aims to transform raw customer data into actional business insights using machine learning techniques based on the Sample Super store dataset from Kaggle.

Features:
Customer Segmentation: Segment statistics and Business Metrics with RFM Analysis- Recency, Frequency and Monegary value segmentation. Average statistics on order value, Product Diversity, Unique Products, and Median Unique Products, order value, and Product diveristy. There is also a customer look up too along with a 2d cluster visualization.
The Market Basket Analysis consists of the Association Rules explorer, our Product Recommendatin engine, Rule visualization and a network graph of top rules. 
The third tab contains the Business Intelligence summary which captures executive summary and highlights along with the recommendations and ROI.

How to Run: Install streamlit and the requirements. txt. Start the Streamlit app:streamlit run app.py. Interact: Enter Customer Analytics Dashboard. Explore the different tabs.

Files: app.py — Streamlit app. Project2_Q3.ipynb — Jupyter notebook with association_rules.csv, cluster_profiles.csv, customer_segments.csv — Saved artifacts along with requirements.txt — Python dependencies.

Requirements: Python 3.8+. See requirements.txt for all packages (pandas, numpy, scikit-learn, streamlit, plotly, mlxtend, scipy and networkx).

Usage Notes: All preprocessing steps and encoders are saved for reproducible predictions. Visualizations are sized for clarity and compactness in the app. Only top features are shown for user input and analysis. No automatic updates—manual edits required for further changes.
