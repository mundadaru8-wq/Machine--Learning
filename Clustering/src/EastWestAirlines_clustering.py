# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:15:29 2026

@author: Ruchi
"""
"""
Business Problem:  
Airline companies need to segment customers based on their flying and loyalty 
program usage patterns. Proper segmentation helps identify high-value customers, 
disengaged customers, and those with unique behaviors like frequent flyers 
who don’t use credit cards. Without such insights, it becomes difficult to 
design targeted loyalty programs, improve customer satisfaction, and prevent churn.

Business Objective:  
Minimize: Customer churn and disengagement by identifying inactive or 
low-engagement flyers and offering reactivation strategies.  
Maximize: Customer lifetime value and loyalty by focusing on high-value 
customers with tailored elite programs, upgrades, and exclusive offers.

Business Constraints: Limited data coverage, overlapping clusters due to 
customer behavior similarities, and moderate clustering quality as shown by 
metrics.
"""
import pandas as pd
import matplotlib.pyplot as plt
#load file 
#df1 = pd.read_excel('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/EastWestAirlines.xlsx')
df1 = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/EastWestAirlines_prep.csv')
a = df1.describe()
#drop useless cols
df = df1.drop(['ID#', 'Award?'], axis=1)
#scale data to remove diff. scale, use normalization or stamdardization
#use normalization for mixed data
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#apply norm function to df
df_norm=norm_func(df)
#check df norm which is scaled
b=df_norm.describe()
#before clustering use dendogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage fu. gives hierarchical or agglomerative clustering
#ref help for linkage
z = linkage(df_norm,method='complete',metric='euclidean')
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
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',
                                   metric='euclidean').fit(df_norm)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this serires to Univ df as col & name col as clust
df['clust']=cluster_labels
#relocate col from 7 to 0 pos.
df1=df.iloc[:,[8,1,2,3,4,5,6,7]]
#check df1
df1.iloc[:,2:].groupby(df1.clust).mean()
#from OP cluster 2 has highest top 10
#lowest acceptance rate, best faculty ratio & highest expenses
#highest graduates ratio
'''
Cluster 0: Active flyers with low credit card usage
- Average enrollment time and miles earned
- Moderate account balance
- Low credit card miles but high flight activity
- Slightly above-average bonus transactions
Insight: These users fly frequently but don’t use credit card miles much. Likely loyal travelers who prefer direct flight benefits over card-based rewards.

Cluster 1: Moderate users with balanced behavior
- Slightly below-average enrollment time and miles
- Average balance and moderate credit card miles
- Moderate bonus transactions, low flight activity
Insight: These are mid-level users with balanced engagement. Could be targeted with offers to increase flight frequency or card usage.

Cluster 2: High-value, high-engagement customers
- Longest enrollment time and highest balance
- Very high credit card miles, bonus transactions, and flight activity
- Above-average qualified miles
Insight: Top-tier customers with full engagement across all channels. Ideal for elite loyalty programs, upgrades, and retention focus.

Cluster 3: Low-engagement, inactive users
- Short enrollment time and lowest balance
- Very low credit card miles, bonus transactions, and flight activity
Insight: These are inactive or disengaged users. May need reactivation campaigns or basic incentives to re-engage.

'''
#clustering performance 
#techniques explaination in notebook
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(df_norm, h_complete.labels_)
db_index = davies_bouldin_score(df_norm, h_complete.labels_)
ch_index = calinski_harabasz_score(df_norm, h_complete.labels_)

print(f'silhouette_score : {silhouette: .4f} (range :-1 to +1, higher is better)')
#0.2597
'''
    score(0.2597):
        low to mid clustering quality.
        thewre is same struct., but clusters overlap or are not well separated.
        possibly , no. of clusters(3)nmis not optimal.
'''
print(f'davies_bouldin_score : {db_index: .4f} (lower is better)') 
#1.1918
'''
score(1.1918):
avg. clustering perf.
clusters are not very tight or well-separated, but they are not completely bad either.
could be improved by:
trying diff. linkage methods (avg, ward, etc)
trying diff. no. od clusters.
'''    
print(f'calinski_harabasz_score : {ch_index: .4f} (higher is better)')    
#1844.7850
'''
your score(1844.7850):

very strong clustering struct.

great grouping with strong separation.

Benefits/Impact of the Solution:  
1. Identifies loyal frequent flyers who rely more on direct benefits than card-based rewards, enabling targeted non-card offers.  
2. Helps design campaigns for mid-level users to increase either flight activity or credit card engagement.  
3. Enables strong retention and personalized strategies for top-tier customers with high engagement across all channels.  
4. Highlights disengaged users for reactivation campaigns and basic incentive programs.  
5. Improves allocation of marketing resources by segmenting based on actual usage behavior rather than generic customer profiles.  
6. Provides measurable clustering structure with a high Calinski-Harabasz score (1844.7850), though moderate silhouette score (0.2597) and Davies-Bouldin score (1.1918) indicate overlapping clusters.  
7. Supports continuous improvement by experimenting with linkage methods and optimal cluster numbers to refine customer groups.  
'''     