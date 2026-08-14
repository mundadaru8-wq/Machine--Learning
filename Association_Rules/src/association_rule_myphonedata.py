# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:30:49 2026

@author: Ruchi
"""

'''
Business Problem:
A film distribution company wants to better target audiences based on their 
color preferences, which may influence visual themes and marketing aesthetics 
in movie promotions. With access to viewer data indicating liked colors, the 
company seeks to uncover patterns in color choices. Association Rule Mining 
helps identify frequent combinations of preferred colors to inform targeted 
campaigns, design decisions, and personalized recommendations.

Business Objective:
Minimize: Ineffective visual marketing and missed engagement by identifying 
strong associations between color preferences  
Maximize: Viewer satisfaction, campaign effectiveness, and brand resonance 
through personalized visual themes and targeted outreach

Business Constraints: Small dataset size, subjective nature of color 
preferences, and limited resources for implementing customized visual 
strategies across diverse audience segments.
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------
# 1. Dataset
# --------------------------

df = pd.read_csv('C:/5-python_crisp-ml(q)/association_rules_py/myphonedata.csv')

print("Data:\n", df)

# --------------------------
# 2. Generate Frequent Itemsets
# --------------------------
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:\n", frequent_itemsets)
'''
Frequent Itemsets:
     support       itemsets
0  0.545455          (red)
1  0.636364        (white)
2  0.545455         (blue)
3  0.363636   (white, red)
4  0.363636    (red, blue)
5  0.363636  (white, blue)

White (support = 0.64), Red (0.55), and Blue (0.55) are the most 
frequently chosen individual colors in this dataset. Based on their 
popularity, we interpret two key associations:

1.White and Red appear together in 36% of cases, suggesting a strong 
co-selection pattern—possibly reflecting aesthetic or thematic pairing 
preferences.

2.Red and Blue co-occur in 36% of cases, indicating a consistent 
combination trend, potentially useful for designing color palettes 
or bundled offerings.
'''
# --------------------------
# 3. Generate Association Rules
# --------------------------
#Rule1 for red
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("red" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
| **Rule (Factor → Outcome)**            | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                                  | **Action Item**                                                                 |
|----------------------------------------|-------------|----------------|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| white → red                            | 0.36        | 0.5714         | 1.05     | 57% of regions where White phones sold well also saw strong Red phone sales—mild co-purchase trend.  | Launch Red model alongside White in regions with high White sales.              |
| blue → red                             | 0.36        | 0.6667         | 1.22     | Two-thirds of Blue phone buyers also chose Red—indicating stronger affinity between these models.    | Bundle Red and Blue models or co-promote them in high Blue-performing regions.  |
'''
#Rule2 for white
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("white" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
| **Rule (Factor → Outcome)**            | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                                  | **Action Item**                                                                 |
|----------------------------------------|-------------|----------------|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| red → white                            | 0.36        | 0.6667         | 1.05     | Two-thirds of Red phone buyers also purchased White—suggesting moderate co-interest.                 | Promote White model in regions with strong Red sales; consider combo offers.     |
| blue → white                           | 0.36        | 0.6667         | 1.05     | 66% of Blue phone buyers also opted for White—indicating consistent cross-preference.                | Co-market White model with Blue in overlapping regions to boost adoption.        |
'''
#Rule3 for blue
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("blue" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
| **Rule (Factor → Outcome)**            | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                                  | **Action Item**                                                                 |
|----------------------------------------|-------------|----------------|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| red → blue                             | 0.36        | 0.6667         | 1.22     | Two-thirds of Red phone buyers also purchased Blue—indicating strong complementary interest.         | Pair Blue model with Red in marketing bundles; target regions with high Red sales. |
| white → blue                           | 0.36        | 0.5714         | 1.05     | 57% of White phone buyers also opted for Blue—suggesting mild cross-preference.                      | Promote Blue model in White-performing regions; test combo offers.               |

Benefits/Impact of the Solution:
1. Finds which colors are liked together by viewers  
2. Helps design better color themes for movies and ads  
3. Suggests color combos for phone or product launches  
4. Improves targeting in visual campaigns  
5. Boosts viewer interest with appealing color choices  
6. Helps plan bundles based on popular color pairs  
'''
