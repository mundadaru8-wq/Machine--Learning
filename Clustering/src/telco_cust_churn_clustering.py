# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:21:48 2026

@author: Ruchi
"""

"""

Business Problem:
Telecom providers need to understand customer usage and service adoption 
patterns to address satisfaction issues and prevent churn. Clustering 
segments customers into meaningful groups for targeted plans, pricing, 
and retention actions.

Business Objective:
Minimize: Churn rate (churning implies customers going to another company 
                      for their needs)  
Maximize: Customer satisfaction (satisfaction will make customer more loyal 
                                 to the brand)

Business Constraints: Lack of data coverage for all customers.
"""
import pandas as pd
import matplotlib.pyplot as plt
#load file 
df = pd.read_excel('C:/5-python_crisp-ml(q)/clustering_datasets/telco_cust_churn_prep.xlsx')
a = df.describe()
df.dtypes
#before clustering use dendogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage fu. gives hierarchical or agglomerative clustering
#ref help for linkage
z = linkage(df,method='complete',metric='euclidean')
plt.figure(figsize=(15,8));
plt.title('hierarchical clustering dendogram')
plt.xlabel('Index');
plt.ylabel('Distance');
#ref help of dendogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=40,leaf_font_size=10)
plt.show()
#dendrogram()
#thumb rule for cluster dicision : elbow method 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Select only numerical features
X = df.select_dtypes(include=['int64', 'float64'])

# Step 2: Compute WCSS (within-cluster sum of squares) for different k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Step 3: Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=5,linkage='complete',
                                   metric='euclidean').fit(df)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this serires to Univ df as col & name col as clust
df['clust']=cluster_labels

# Move last column to first
cols = df.columns.tolist()
new_order = [cols[-1]] + cols[:-1]
df1 = df[new_order]
#check df1
df1.iloc[:,2:].groupby(df1.clust).mean()
'''
Cluster 0: Full-service, high-engagement users
- Long tenure and full phone + internet service
- High usage of all add-ons (security, backup, streaming, tech support)
- High monthly charges and revenue
- Mostly on fiber optic, with one- or two-year contracts
Insight: These are loyal, high-paying customers who use nearly every service. Ideal for retention programs and bundled premium offers.

Cluster 1: Basic users with low revenue
- Short tenure and low total charges
- Moderate internet usage, fewer add-ons
- Low extra data charges and refunds
- Mostly on DSL or fiber, with one-year contracts
Insight: Low-value segment with minimal engagement. May respond to upsell offers or incentives to adopt more services.

Cluster 2: Phone-only, no internet users
- Moderate tenure, full phone service
- No internet or add-ons at all
- Very low total charges and revenue
- All on two-year contracts
Insight: These are legacy or minimal users. Likely not interested in digital services. May be hard to upsell but stable.

Cluster 3: Mid-tier internet users with high data usage
- Moderate tenure and internet service
- Some add-ons (security, backup, streaming)
- Very high monthly data usage
- Low total revenue and charges
Insight: Heavy internet users with average spend. Could be targeted with data-focused plans or streaming bundles.

Cluster 4: Premium internet users with high spend
- Long tenure and full service (phone + internet)
- High usage of all add-ons and streaming
- Highest monthly charges, refunds, and data usage
- High total revenue and referrals
Insight: Top-tier customers with full engagement. Ideal for loyalty rewards, exclusive offers, and retention focus.
'''
#clustering performance 
#techniques explaination in notebook
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(df, h_complete.labels_)
db_index = davies_bouldin_score(df, h_complete.labels_)
ch_index = calinski_harabasz_score(df, h_complete.labels_)

print(f'silhouette_score : {silhouette: .4f} (range :-1 to +1, higher is better)')
#0.2271
'''
    score(0.2271):
    low clustering quality.
    clusters are weakly separated and may overlap significantly.
    structure is minimal — data points may not form distinct groups.

'''
print(f'davies_bouldin_score : {db_index: .4f} (lower is better)') 
#1.6165
'''
score(1.6165):
    above-average clustering performance.
    clusters show noticeable separation and structure, though not perfect.
    some overlap may still exist, but overall grouping is meaningful.
    could be refined by:
        testing different linkage methods (e.g., ward, complete)
        validating cluster count with silhouette or elbow method
'''    
print(f'calinski_harabasz_score : {ch_index: .4f} (higher is better)')    
#1169.0343
'''
score(1169.0343):
    strong clustering structure.
    clearly better than random or trivial grouping.
    clusters show meaningful separation and internal consistency.
    still, refinement is possible — consider tuning features or testing alternate algorithms.

Benefits/Impact of the Solution:
1. Enables targeted offers and plan bundles tailored to each segment
2. Improves retention by prioritizing at-risk customer groups
3. Increases revenue through cross-sell and up-sell of add-ons and premium support
4. Optimizes support operations by aligning service levels with segment needs
5. Reduces refunds and complaints by focusing quality improvements where they matter
6. Enhances campaign ROI with data-driven audience selection and measurement
'''     
    