# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:18:32 2026

@author: Ruchi
"""
"""
Business Problem:
Auto insurance companies need to understand customer profiles and behavior 
to design better policies, prevent churn, and increase revenue. Clustering 
helps in segmenting customers into meaningful groups for targeted strategies.

Business Objective:
Minimize: Churn rate (reduce the risk of customers leaving for competitors)  
Maximize: Customer satisfaction (retain high-value clients through p
                                 ersonalized services and offers)

Business Constraints: 
    Lack of complete customer behavior data, 
    changing customer preferences over time, 
    and operational feasibility of applying differentiated strategies.
"""
import pandas as pd
import matplotlib.pyplot as plt
#load file 
df = pd.read_csv('C:/5-python_crisp-ml(q)/clustering_datasets/auto_insurance_prep.csv')
a = df.describe()
df.dtypes
#before clustering use dendogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage fu. gives hierarchical or agglomerative clustering
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

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=6,linkage='complete',
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
Cluster 0: Urban, mid-risk customers
- High coverage and moderately high claim amounts
- Mostly live in urban areas
- Prefer personal auto policies and buy via web
- Moderate number of policies and complaints
Insight: These are typical urban customers with decent coverage and digital habits. Good candidates for online retention offers and mid-tier policy upgrades.

Cluster 1: High-complaint, multi-policy users
- Very high number of complaints and multiple policies
- Average income and policy duration
- Mostly personal auto users, active on web
Insight: These customers raise frequent issues and manage several policies. They need better support and possibly segmented handling to reduce churn.

Cluster 2: High-value suburban clients
- Highest coverage and claim amounts
- Strong suburban presence
- High lifetime value and premium payments
- Low complaints, moderate policy count
Insight: Loyal and profitable customers. Ideal for premium plans, long-term retention, and personalized service.

Cluster 3: Low-claim, high-policy volume
- Most policies but lowest claim amounts
- Stable employment and marital status
- Mostly personal auto, prefer web channel
Insight: Likely bundled or corporate accounts with low risk. Focus on engagement and cross-selling opportunities.

Cluster 4: Premium suburban spenders
- High coverage, claim amount, and number of policies
- Strong suburban presence
- High lifetime value and premium
Insight: Similar to Cluster 2 but with even more spending. Target with elite-tier services and loyalty programs.

Cluster 5: Low-risk, low-engagement users
- Low coverage, claim amount, and policy count
- Few complaints, moderate web usage
- Balanced demographics
Insight: Passive customers with low activity. Use incentives and personalized nudges to increase engagement or upsell.
'''
#clustering performance 
#techniques explaination in notebook
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(df, h_complete.labels_)
db_index = davies_bouldin_score(df, h_complete.labels_)
ch_index = calinski_harabasz_score(df, h_complete.labels_)

print(f'silhouette_score : {silhouette: .4f} (range :-1 to +1, higher is better)')
#0.1790
'''
    score(0.1790):
    weak clustering quality.
    minimal structure detected — clusters likely overlap heavily.
    separation between groups is poor, and boundaries may be unclear.
    likely that the number of clusters is suboptimal — test alternatives like 2, 4.
'''
print(f'davies_bouldin_score : {db_index: .4f} (lower is better)') 
#2.2613
'''
score(2.2613):
    good clustering performance.
    clusters show clear separation and meaningful structure, though not perfect.
    some overlap may still exist, but overall grouping is strong.
    could be improved by:
        testing different linkage methods (e.g., ward, complete)
        validating cluster count with elbow method
'''    
print(f'calinski_harabasz_score : {ch_index: .4f} (higher is better)')    
#1332.2019
'''
range : 0 to infinity

interpret:
maesures ratio of between-cluster dispersion to within-cluster cdispersion.

higher values are better - they indicate well-separated, compact clusters.

no universal scale, but values :
> 100 = strong separation

10-50 = moderate separation

< 10 = weak

score(1332.2019):
    moderately strong clustering structure.
    clearly better than random or trivial grouping.
    clusters show some separation and internal consistency, but not optimal.

Benefits/Impact of the Solution:
1. Enables targeted marketing campaigns for different customer groups  
2. Improves customer retention by identifying at-risk segments  
3. Supports revenue growth through cross-selling and up-selling opportunities  
4. Enhances profitability by differentiating low-risk and high-risk customers  
5. Helps allocate resources more efficiently across customer segments  
6. Provides actionable insights for strategic decision-making
'''     
    
