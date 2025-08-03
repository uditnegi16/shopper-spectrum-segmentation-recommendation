# streamlit_app.py

import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# =======================
# Load & Prepare Data
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("../data/online_retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# =======================
# RFM Feature Engineering
# =======================
def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# =======================
# Clustering Model Loader
# =======================
def load_models():
    kmeans = pickle.load(open("../app/model/kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("../app/model/scaler.pkl", "rb"))
    return kmeans, scaler

# =======================
# Predict Cluster Label
# =======================
def predict_cluster(r, f, m, kmeans, scaler):
    scaled = scaler.transform([[r, f, m]])
    cluster = kmeans.predict(scaled)[0]
    labels = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }
    return labels.get(cluster, f"Cluster {cluster}")

# =======================
# Product Recommendation
# =======================
def build_similarity_matrix(df):
    pivot = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum', fill_value=0)
    similarity = cosine_similarity(pivot.T)
    similarity_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)
    return similarity_df

def recommend(product_code, sim_df, n=5):
    if product_code not in sim_df.columns:
        return []
    return sim_df[product_code].sort_values(ascending=False).iloc[1:n+1].index.tolist()

# =======================
# Streamlit UI
# =======================
def main():
    st.set_page_config(page_title="🛒 Shopper Spectrum", layout="centered")
    st.title("🛍️ Shopper Spectrum - E-Commerce Analytics App")

    df = load_data()
    rfm = compute_rfm(df)
    kmeans, scaler = load_models()
    sim_df = build_similarity_matrix(df)

    tab1, tab2 = st.tabs(["📦 Product Recommendations", "👤 Customer Segmentation"])

    from difflib import get_close_matches  # Place this at the top of the file

    # ----------------------------------------
    # 📦 Product Recommendations by Name
    # ----------------------------------------
    with tab1:
        st.subheader("🔍 Find Similar Products by Name")

        # Get unique product names from dataset
        product_names = sorted(df['Description'].dropna().unique().tolist())
        user_input = st.text_input("Enter Product Name (e.g. WHITE HANGING HEART T-LIGHT HOLDER):")

        if st.button("Get Recommendations"):
            # Find closest match (case-insensitive)
            close_matches = get_close_matches(user_input.upper(), product_names, n=1, cutoff=0.6)
            if close_matches:
                matched_name = close_matches[0]
                st.info(f"Closest match found: **{matched_name}**")
                match_row = df[df['Description'] == matched_name].iloc[0]
                product_code = match_row['StockCode']

                recommendations = recommend(product_code, sim_df)
                if recommendations:
                    st.success("Top 5 similar products:")
                    for i, rec in enumerate(recommendations, 1):
                        # Optional: Show product names instead of codes
                        name = df[df['StockCode'] == rec]['Description'].dropna().unique()
                        name_str = name[0] if len(name) > 0 else rec
                        st.markdown(f"**{i}.** `{rec}` — {name_str}")
                else:
                    st.warning("No similar products found.")
            else:
                st.error("❌ No close match found. Try another product name.")


    with tab2:
        st.subheader("🧠 Predict Customer Segment")
        r = st.number_input("Recency (days since last purchase):", min_value=0)
        f = st.number_input("Frequency (number of purchases):", min_value=0)
        m = st.number_input("Monetary (total spend):", min_value=0.0)

        if st.button("Predict Cluster"):
            label = predict_cluster(r, f, m, kmeans, scaler)
            st.success(f"📌 Predicted Segment: **{label}**")

if __name__ == "__main__":
    main()
