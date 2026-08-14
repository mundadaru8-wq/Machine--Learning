# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:23:13 2026

@author: Ruchi
"""

Business Problem:
Kitabi Duniya, a heritage bookstore in India, has experienced a decline in 
annual growth due to the rise of online book sales and widespread Internet 
access. To regain popularity and increase customer footfall, the store needs 
to uncover patterns in customer purchasing behavior. Association Rule Mining 
can help identify frequently co-purchased book categories to inform bundling, 
store layout, and promotional strategies.

Business Objective:
Minimize: Missed cross-selling opportunities and unsold inventory by identifying 
frequent itemsets and customer preferences  
Maximize: Customer engagement, footfall, and revenue through targeted promotions 
and personalized recommendations

Business Constraints: Sparse and incomplete transaction data due to manual 
record-keeping, limited digital infrastructure, and constrained marketing 
resources for implementing dynamic bundling or personalized campaigns.
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------
# 1. Dataset
# --------------------------

df = pd.read_csv('C:/5-python_crisp-ml(q)/association_rules_py/book.csv')

print("Data:\n", df)

# --------------------------
# 2. Generate Frequent Itemsets
# --------------------------
frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)

print("\nFrequent Itemsets:\n", frequent_itemsets)
'''
Frequent Itemsets:
    support    itemsets
0    0.423  (ChildBks)
1    0.431   (CookBks)

ChildBks and CookBks are the most frequently purchased individual book 
categories so we will interpret 2 assoiciation rules based on these 2 book 
categories.
'''
# --------------------------
# 3. Generate Association Rules
# --------------------------
#Rule1 for ChildBks
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Filter rules where ChildBks is the consequent
rules = rules[rules['consequents'].apply(lambda x: any("ChildBks" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
Association Rules :
   antecedents consequents  support  confidence      lift
0  (YouthBks)  (ChildBks)   0.1650    0.666667  1.576044
2  (DoItYBks)  (ChildBks)   0.1840    0.652482  1.542511
3    (RefBks)  (ChildBks)   0.1515    0.706294  1.669725
4    (ArtBks)  (ChildBks)   0.1625    0.674274  1.594028
5   (GeogBks)  (ChildBks)   0.1950    0.706522  1.670264

| **Rule (Factor → Outcome)**              | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                 | **Action Item**                                                  |
| ---------------------------------------- | ----------- | -------------- | -------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| YouthBks → ChildBks                      | 0.1650      | 0.6667         | 1.58     | Youth books are often bought alongside children's books—66% chance of co-purchase. | Bundle youth and children's books in promotions or recommendations. |
| DoItYBks → ChildBks                      | 0.1840      | 0.6525         | 1.54     | DIY books frequently co-occur with children's books—65% chance.                    | Cross-market DIY kits with children's books for creative engagement. |
| RefBks → ChildBks                        | 0.1515      | 0.7063         | 1.67     | Reference books strongly predict children's book purchases—71% chance.             | Position reference and children's books together in catalog or shelf. |
| ArtBks → ChildBks                        | 0.1625      | 0.6743         | 1.59     | Art books are often bought with children's books—67% chance.                       | Promote art-themed children's books or joint learning bundles.       |
| GeogBks → ChildBks                       | 0.1950      | 0.7065         | 1.67     | Geography books strongly co-occur with children's books—71% chance.                | Design educational bundles combining geography and children's content. |

'''
#Rule2 for CookBks
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Filter rules where CookBks is the consequent
rules = rules[rules['consequents'].apply(lambda x: any("CookBks" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
Association Rules :
    antecedents consequents  support  confidence      lift
1   (ChildBks)   (CookBks)   0.2560    0.605201  1.404179
6   (YouthBks)   (CookBks)   0.1620    0.654545  1.518667
7   (DoItYBks)   (CookBks)   0.1875    0.664894  1.542677
8     (RefBks)   (CookBks)   0.1525    0.710956  1.649549
9     (ArtBks)   (CookBks)   0.1670    0.692946  1.607763
10   (GeogBks)   (CookBks)   0.1925    0.697464  1.618245

| **Rule (Factor → Outcome)**              | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                 | **Action Item**                                                  |
| ---------------------------------------- | ----------- | -------------- | -------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| ChildBks → CookBks                       | 0.2560      | 0.6052         | 1.40     | 61% of ChildBks buyers also purchase CookBks—moderate co-purchase tendency.        | Recommend CookBks to ChildBks buyers via upsell or cross-sell.   |
| YouthBks → CookBks                       | 0.1620      | 0.6545         | 1.52     | YouthBks buyers are 1.5× more likely to buy CookBks—stronger association.          | Bundle youth and cooking books for teen engagement.              |
| DoItYBks → CookBks                       | 0.1875      | 0.6649         | 1.54     | DIY book buyers often buy CookBks—66% chance, with 1.5× lift.                      | Promote DIY + cooking kits for hands-on learning.                |
| RefBks → CookBks                         | 0.1525      | 0.7110         | 1.65     | Reference book buyers have a 71% chance of buying CookBks—strongest predictor.     | Position CookBks near reference sections or educational bundles. |
| ArtBks → CookBks                         | 0.1670      | 0.6929         | 1.61     | Art book buyers frequently buy CookBks—69% chance, high lift.                      | Curate aesthetic cooking collections for art-focused readers.    |
| GeogBks → CookBks                        | 0.1925      | 0.6975         | 1.62     | Geography book buyers are 1.6× more likely to buy CookBks—70% chance.              | Create travel-themed cooking bundles (e.g., world cuisines).     |
'''
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Create a matrix of item co-occurrence
frequent_itemsets = df.T.dot(df)
sns.heatmap(frequent_itemsets, annot=True, cmap='YlGnBu')
plt.title('Item Co-occurrence Heatmap')
plt.show()
'''
Benefits/Impact of the Solution:
1. Finds which book types are often bought together  
2. Helps create book bundles to boost sales  
3. Improves shelf arrangement based on buying patterns  
4. Suggests books customers may also like  
5. Makes marketing more focused and effective  
6. Increases customer visits and satisfaction  
7. Reduces unsold books by stocking smartly  
8. Shows how data can help grow the business  
'''