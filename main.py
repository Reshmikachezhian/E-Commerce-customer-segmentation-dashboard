import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def safe_qcut(series, q, labels_desc):
    try:
        # Try qcut with labels
        return pd.qcut(series, q=q, labels=labels_desc[:q], duplicates='drop')
    except ValueError:
        # Manual binning fallback
        quantiles = series.quantile([i/q for i in range(q + 1)]).unique()
        num_bins = len(quantiles) - 1
        if num_bins < 1:
            return pd.Series([labels_desc[-1]] * len(series), index=series.index)
        return pd.cut(series, bins=quantiles, labels=labels_desc[:num_bins], include_lowest=True)


def main():
    # Step 1: Load data
    df = pd.read_csv("ecommerce_transactions.csv")

    # Step 2: Parse date correctly
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], dayfirst=True)

    # Step 3: RFM Analysis
    today = df['Transaction_Date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('User_Name').agg({
        'Transaction_Date': lambda x: (today - x.max()).days,   # Recency
        'Transaction_ID': 'count',                              # Frequency
        'Purchase_Amount': 'sum'                                # Monetary
    }).reset_index()

    rfm.columns = ['User_Name', 'Recency', 'Frequency', 'Monetary']

    # Step 4: RFM scoring with safe dynamic binning
    rfm['R_Score'] = safe_qcut(rfm['Recency'], 5, [5, 4, 3, 2, 1])
    rfm['F_Score'] = safe_qcut(rfm['Frequency'].rank(method="first"), 5, [1, 2, 3, 4, 5])
    rfm['M_Score'] = safe_qcut(rfm['Monetary'], 5, [1, 2, 3, 4, 5])

    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Step 5: Merge RFM back into main dataset
    df = df.merge(rfm, on='User_Name', how='left')

    # Step 6: Drop non-numeric/categorical columns for clustering
    df_clean = df.drop(columns=[
        'Transaction_ID', 'User_Name', 'Transaction_Date',
        'R_Score', 'F_Score', 'M_Score', 'RFM_Score'
    ], errors='ignore')

    # Step 7: Label encode categorical columns
    categorical_cols = df_clean.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Step 8: Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # Step 9: Elbow Method to find optimal k
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Curve
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

    # Step 10: Apply KMeans with chosen k (adjust as needed)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Step 11: Save output to CSV
    df.to_csv("clustered_with_rfm.csv", index=False)
    print("\n✅ Clustered data with RFM saved to 'clustered_with_rfm.csv'.")

if __name__ == "__main__":
    main()
