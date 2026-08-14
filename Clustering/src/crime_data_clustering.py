# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:20:20 2026

@author: Ruchi
"""
"""

Business Problem:
Government and law enforcement agencies need to analyze crime patterns across 
regions to identify high-risk zones, allocate resources efficiently, and 
improve public safety. Clustering helps in segmenting regions into meaningful 
groups based on crime and population data.

Business Objective:
Minimize: Crime incidents by proactively targeting high-risk clusters with 
interventions  
Maximize: Public safety and efficient use of policing resources

Business Constraints: Limited availability of complete and consistent crime 
data across all states, differences in reporting standards, and resource 
limitations for implementation.
"""
import pandas as pd
import matplotlib.pyplot as plt
#load file 
#df1 = pd.read_excel('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/EastWestAirlines.xlsx')
df1 = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/crime_data_prep.csv')
a = df1.describe()
#drop useless cols
df = df1.drop(['State'], axis=1)
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
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',
                                   metric='euclidean').fit(df)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this serires to Univ df as col & name col as clust
df['clust']=cluster_labels
#relocate col from 7 to 0 pos.
df1=df.iloc[:,[4,0,1,2,3]]
#check df1
df1.iloc[:,2:].groupby(df1.clust).mean()
'''
Cluster 0: High-risk urban zones
 Highest assault rates (0.87 std above mean)
 Average urban population
 Elevated rape incidence
Inference: These regions likely represent densely populated, high-crime urban 
areas. Intervention and policing resources may be prioritized here.

Cluster 1: Moderately urbanized, mixed-risk
 Slightly below-average assault
 Above-average urban population
 Near-average rape rates
Inference: These areas may be urban but relatively safer, possibly benefiting 
from better infrastructure or community programs.

Cluster 2: Low-crime, rural zones
 Lowest assault rates
 Least urbanized
 Lowest rape incidence
Inference: These regions likely represent rural or suburban communities with 
lower crime exposure. May have different socio-economic dynamics.

'''
#clustering performance 
#techniques explaination in notebook
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(df, h_complete.labels_)
db_index = davies_bouldin_score(df, h_complete.labels_)
ch_index = calinski_harabasz_score(df, h_complete.labels_)

print(f'silhouette_score : {silhouette: .4f} (range :-1 to +1, higher is better)')
#0.3637
'''
    score(0.3637):
    moderate clustering quality.
    some structure is present, but clusters may overlap or lack clear separation.
    possibly, number of clusters (3) is not optimal — consider testing 2 or 4.

'''
print(f'davies_bouldin_score : {db_index: .4f} (lower is better)') 
#0.9656
'''
score(0.9656):
    average clustering performance.
    clusters are not very tight or well-separated, but they’re not completely poor either.
    could be improved by:
        trying different linkage methods (e.g., average, ward)
        experimenting with different numbers of clusters

'''    
print(f'calinski_harabasz_score : {ch_index: .4f} (higher is better)')    
#39.2064
'''
your score(39.2064):

moderate clustering struct.

not great, but better than random or trivial grouping.

again suggests some separation, but there is room to improve

Benefits/Impact of the Solution:
1. Identifies high-risk urban clusters requiring immediate policing and preventive actions  
2. Helps design targeted policies and community programs for moderately urbanized areas  
3. Highlights low-crime rural clusters for monitoring and preventive strategies instead of heavy policing  
4. Enables data-driven allocation of law enforcement resources across regions  
5. Provides insights for urban planning and socio-economic development policies  
6. Supports long-term crime reduction by identifying structural patterns behind crime rates
'''     
    