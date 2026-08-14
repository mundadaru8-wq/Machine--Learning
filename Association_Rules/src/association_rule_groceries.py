# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:27:28 2026

@author: Ruchi
"""

'''
Business Problem:
The Departmental Store has collected daily transaction data of products sold. 
To optimize product placement, improve cross-selling strategies, and enhance 
customer shopping experience, the store needs to identify frequent item 
combinations. Association Rule Mining helps uncover patterns in customer 
purchases to drive smarter inventory and marketing decisions.

Business Objective:
Minimize: Missed bundling opportunities and inefficient shelf arrangements by 
identifying frequently co-purchased items  
Maximize: Sales revenue and customer satisfaction through targeted promotions 
and strategic product placement

Business Constraints: Limited transaction volume, variability in customer 
preferences, and resource constraints for implementing dynamic marketing and 
inventory changes across departments.
'''

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------
# 1. Dataset
# --------------------------

with open('C:/5-python_crisp-ml(q)/association_rules_py/groceries.csv', 'r') as f:
    lines = f.readlines()

transactions = [line.strip().split(',') for line in lines]

print("Data:\n", transactions)

# --------------------------
# 2. Transaction Encoding
# --------------------------
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)


print("\nOne-Hot Encoded Data:\n", df.head())

# --------------------------
# 3. Generate Frequent Itemsets
# --------------------------
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

#sort in descending order
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print("\nFrequent Itemsets:\n", frequent_itemsets)
'''
Frequent Itemsets:
      support                        itemsets
26  0.255516                    (whole milk)
15  0.193493              (other vegetables)
19  0.183935                    (rolls/buns)
23  0.174377                          (soda)
27  0.139502                        (yogurt)

above mentioned categories of food are top 5 most purchased item from the 
grocery store.

support
0.074835  (other vegetables, whole milk)
0.056634        (whole milk, rolls/buns)
0.056024            (whole milk, yogurt)

above are top 3 most frequently bought together items.
'''
# --------------------------
# 4. Generate Association Rules
# --------------------------
#Rule1 for whole milk
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("whole milk" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
Association Rules :
           antecedents   consequents   support  confidence      lift
0  (other vegetables)  (whole milk)  0.074835    0.386758  1.513634
1        (rolls/buns)  (whole milk)  0.056634    0.307905  1.205032
2            (yogurt)  (whole milk)  0.056024    0.401603  1.571735

| **Rule (Factor → Outcome)**                      | **Support** | **Confidence** | **Lift** | **Interpretation**                                                       | **Action Item**                                                   |
|--------------------------------------------------|-------------|----------------|----------|----------------------------------------------------------------------------|-------------------------------------------------------------------|
| other_vegetables → whole_milk                   | 0.0748      | 0.3868         | 1.5136   | Buying vegetables increases chance of buying whole milk by 1.5×.          | Cross-promote milk with fresh produce.                           |
| rolls_buns → whole_milk                         | 0.0566      | 0.3079         | 1.2050   | Customers buying buns are 20% more likely to buy whole milk.              | Bundle bakery items with dairy offers.                           |
| yogurt → whole_milk                             | 0.0560      | 0.4016         | 1.5717   | Yogurt buyers show strong affinity toward whole milk (40% confidence).    | Position yogurt and milk together in dairy aisle.                |
'''
#Rule2 for other vegetables
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("other vegetables" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
Association Rules :
     antecedents         consequents   support  confidence      lift
1  (whole milk)  (other vegetables)  0.074835    0.292877  1.513634

| **Rule (Factor → Outcome)**            | **Support** | **Confidence** | **Lift** | **Interpretation**                                                        | **Action Item**                                                  |
|----------------------------------------|-------------|----------------|----------|-----------------------------------------------------------------------------|------------------------------------------------------------------|
| whole_milk → other_vegetables          | 0.0748      | 0.2929         | 1.5136   | Customers buying whole milk are 1.5× more likely to also buy vegetables.   | Promote fresh produce near dairy section to boost cross-sales.  |
'''
#Rule for item pairs
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])

'''
Association Rules :
           antecedents         consequents   support  confidence      lift
0  (other vegetables)        (whole milk)  0.074835    0.386758  1.513634
1        (whole milk)  (other vegetables)  0.074835    0.292877  1.513634
2        (whole milk)        (rolls/buns)  0.056634    0.221647  1.205032
3        (rolls/buns)        (whole milk)  0.056634    0.307905  1.205032
4        (whole milk)            (yogurt)  0.056024    0.219260  1.571735
5            (yogurt)        (whole milk)  0.056024    0.401603  1.571735

| **Rule (Factor → Outcome)**            | **Support** | **Confidence** | **Lift** | **Interpretation**                                                        | **Action Item**                                                  |
|----------------------------------------|-------------|----------------|----------|-----------------------------------------------------------------------------|------------------------------------------------------------------|
| other_vegetables → whole_milk          | 0.0748      | 0.3868         | 1.5136   | Vegetable buyers are 1.5× more likely to also buy whole milk.              | Promote milk near fresh produce to boost joint sales.            |
| whole_milk → other_vegetables          | 0.0748      | 0.2929         | 1.5136   | Whole milk buyers often pick up vegetables—29% chance, above average.      | Suggest fresh produce to dairy buyers.                           |
| whole_milk → rolls_buns                | 0.0566      | 0.2216         | 1.2050   | Milk buyers are slightly more likely to buy buns—22% chance.               | Bundle milk with bakery items for breakfast shoppers.            |
| rolls_buns → whole_milk                | 0.0566      | 0.3079         | 1.2050   | Bun buyers have a 31% chance of buying milk—mild association.              | Cross-promote dairy with bakery section.                         |
| whole_milk → yogurt                    | 0.0560      | 0.2193         | 1.5717   | Milk buyers are 1.57× more likely to buy yogurt—strong lift.               | Recommend yogurt to milk buyers in dairy aisle.                  |
| yogurt → whole_milk                    | 0.0560      | 0.4016         | 1.5717   | Yogurt buyers have a 40% chance of buying milk—strongest confidence here.  | Position milk and yogurt together for easy access.               |

Benefits/Impact of the Solution:
1. Shows which grocery items are often bought together  
2. Helps arrange products smartly on shelves  
3. Suggests good combos for offers and bundles  
4. Makes shopping easier and more enjoyable for customers  
5. Boosts sales by promoting popular item pairs  
6. Reduces unsold stock by stocking what people buy together  
7. Supports better planning for inventory and marketing  
8. Uses data to improve store layout and customer experience  
'''


