import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset from Excel
# -----------------------------
df = pd.read_excel('Online Retail.xlsx')

# Keep only positive quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# 2. Create TotalPrice column
# -----------------------------
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# 3. Compute RFM metrics
# -----------------------------
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalPrice': 'sum'                                       # Monetary
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'MonetaryValue'
}, inplace=True)

# -----------------------------
# 4. Assign RFM scores (1-5) robustly
# -----------------------------
# Recency
rfm['R_Score'], r_bins = pd.qcut(rfm['Recency'], 5, labels=False, retbins=True, duplicates='drop')
rfm['R_Score'] = rfm['R_Score'] + 1  # convert 0-based to 1-5

# Frequency
rfm['F_Score'], f_bins = pd.qcut(rfm['Frequency'], 5, labels=False, retbins=True, duplicates='drop')
rfm['F_Score'] = rfm['F_Score'] + 1

# Monetary
rfm['M_Score'], m_bins = pd.qcut(rfm['MonetaryValue'], 5, labels=False, retbins=True, duplicates='drop')
rfm['M_Score'] = rfm['M_Score'] + 1

# Combine RFM score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# -----------------------------
# 5. Define segments based on RFM score
# -----------------------------
def segment_customer(row):
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif f >= 4:
        return 'Loyal'
    elif r <= 2 and f >= 3:
        return 'Potential Loyalist'
    elif r <= 2 and f <= 2:
        return 'At-risk'
    elif r >= 4 and f <= 2 and m <= 2:
        return 'Hibernating'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# -----------------------------
# 6. Scale features for clustering (optional)
# -----------------------------
features = ['Recency','Frequency','MonetaryValue']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# -----------------------------
# 7. Fit K-Means (optional)
# -----------------------------
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Silhouette score
sil_score = silhouette_score(X_scaled, rfm['Cluster'])
print(f"Silhouette Score: {sil_score:.3f}")

# -----------------------------
# 8. Analyze segments
# -----------------------------
segment_summary = rfm.groupby('Segment').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean','count']
}).round(1)

print("\nSegment Summary:")
print(segment_summary)

# -----------------------------
# 9. Visualize segments
# -----------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    x='MonetaryValue',
    y='Frequency',
    hue='Segment',
    palette='Set1',
    data=rfm,
    s=100
)
plt.title('Customer Segmentation (RFM)')
plt.xlabel('Monetary Value')
plt.ylabel('Frequency')
plt.show()