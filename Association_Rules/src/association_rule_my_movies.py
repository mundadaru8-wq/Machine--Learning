# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:29:40 2026

@author: Ruchi
"""

'''
Business Problem:
A film distribution company wants to better target audiences based on their 
movie preferences. With access to viewer data indicating likes and dislikes 
across various films, the company seeks to uncover patterns in movie choices. 
Association Rule Mining helps identify frequent combinations of liked movies 
to inform targeted marketing, bundling strategies, and personalized 
recommendations.

Business Objective:
Minimize: Ineffective promotions and missed audience engagement by identifying 
strong associations between viewer preferences  
Maximize: Audience satisfaction, targeted outreach, and revenue through 
personalized movie bundles and recommendation systems

Business Constraints: Limited dataset size, variability in viewer tastes, and 
resource constraints for implementing personalized marketing and distribution 
strategies across platforms.
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------
# 1. Dataset
# --------------------------

df = pd.read_csv('C:/5-python_crisp-ml(q)/association_rules_py/my_movies.csv')

print("Data:\n", df)

# --------------------------
# 2. Generate Frequent Itemsets
# --------------------------
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:\n", frequent_itemsets)
'''
Frequent Itemsets:
    support                           itemsets
0      0.6                      (Sixth Sense)
1      0.7                        (Gladiator)
2      0.6                          (Patriot)
3      0.5           (Gladiator, Sixth Sense)
4      0.4             (Patriot, Sixth Sense)
5      0.6               (Patriot, Gladiator)
6      0.4  (Patriot, Gladiator, Sixth Sense)

Gladiator (support = 0.7), Sixth Sense (0.6), and Patriot (0.6) are the most 
frequently watched individual movies in this dataset. Based on their popularity,
we’ll interpret two key associations involving these titles:

1. Gladiator and Sixth Sense are watched together by 50% of users, indicating a
strong co-viewing pattern between action and thriller genres.

2. Patriot and Gladiator appear together in 60% of cases, suggesting a high 
overlap in audience preference for historical and war-themed films.
'''
# --------------------------
# 3. Generate Association Rules
# --------------------------
#Rule1 for Sixth Sense
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("Sixth Sense" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
| **Rule (Factor → Outcome)**                        | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                                  | **Action Item**                                                                 |
|----------------------------------------------------|-------------|----------------|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Gladiator → Sixth Sense                            | 0.50        | 0.7143         | 1.19     | 71% of Gladiator viewers also liked Sixth Sense—indicating strong cross-genre appeal.                | Recommend thrillers like Sixth Sense to action movie fans.                      |
| Patriot → Sixth Sense                              | 0.40        | 0.6667         | 1.11     | Two-thirds of Patriot viewers also watched Sixth Sense—suggesting overlap in historical and thriller tastes. | Promote psychological thrillers to war movie audiences.                  |
| Gladiator + Patriot → Sixth Sense                  | 0.40        | 0.6667         | 1.11     | Viewers who liked both Gladiator and Patriot are likely to enjoy Sixth Sense too.                    | Target bundled promotions for fans of action and war genres with thrillers.     |
| Patriot → Gladiator + Sixth Sense                  | 0.40        | 0.6667         | 1.33     | Patriot fans often go on to watch both Gladiator and Sixth Sense—strong multi-genre interest.        | Suggest curated movie packs combining war, action, and thriller themes.         |
| Gladiator → Patriot + Sixth Sense                  | 0.40        | 0.5714         | 1.43     | Over half of Gladiator viewers also liked Patriot and Sixth Sense—high lift across genres.           | Build audience segments for cross-genre marketing: action + war + thriller.     |

'''
#Rule2 for Gladiator
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("Gladiator" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
| **Rule (Factor → Outcome)**                        | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                                  | **Action Item**                                                                 |
|----------------------------------------------------|-------------|----------------|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Sixth Sense → Gladiator                            | 0.50        | 0.8333         | 1.19     | 83% of Sixth Sense viewers also liked Gladiator—strong crossover from thriller to action.             | Recommend action films to psychological thriller audiences.                     |
| Patriot → Gladiator                                | 0.40        | 1.0000         | 1.43     | All Patriot viewers also watched Gladiator—perfect overlap in war and action genres.                  | Prioritize Gladiator in campaigns targeting Patriot fans.                       |
| Patriot + Sixth Sense → Gladiator                  | 0.40        | 1.0000         | 1.43     | Viewers who liked both Patriot and Sixth Sense always watched Gladiator—multi-genre synergy.          | Create bundled recommendations for war + thriller fans with Gladiator as anchor.|
| Patriot → Gladiator + Sixth Sense                  | 0.40        | 0.6667         | 1.33     | Two-thirds of Patriot fans also liked both Gladiator and Sixth Sense—strong multi-genre interest.     | Promote curated packs combining war, action, and thriller themes.               |
| Sixth Sense → Gladiator + Patriot                  | 0.40        | 0.6667         | 1.11     | 67% of Sixth Sense viewers also liked Gladiator and Patriot—suggests layered genre preferences.       | Target thriller audiences with bundled war-action recommendations.              |
'''
#Rule3 for Patriot
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Filter rules 
rules = rules[rules['consequents'].apply(lambda x: any("Patriot" in s for s in x))]

print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
| **Rule (Factor → Outcome)**                        | **Support** | **Confidence** | **Lift** | **Interpretation**                                                                                  | **Action Item**                                                                 |
|----------------------------------------------------|-------------|----------------|----------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Sixth Sense → Patriot                              | 0.40        | 0.6667         | 1.11     | Two-thirds of Sixth Sense viewers also liked Patriot—suggests thriller fans lean toward war dramas.  | Recommend historical war films to psychological thriller audiences.             |
| Gladiator → Patriot                                | 0.40        | 0.8571         | 1.43     | 86% of Gladiator fans also watched Patriot—strong affinity between action and war genres.            | Cross-promote war films to action movie viewers.                                 |
| Gladiator + Sixth Sense → Patriot                  | 0.40        | 0.8000         | 1.33     | 80% of viewers who liked both Gladiator and Sixth Sense also liked Patriot—multi-genre synergy.      | Bundle war films with action-thriller combos for targeted recommendations.       |
| Sixth Sense → Gladiator + Patriot                  | 0.40        | 0.6667         | 1.11     | 67% of Sixth Sense viewers also liked Gladiator and Patriot—layered genre interest.                  | Promote curated packs for thriller fans featuring war and action themes.         |
'''
#Rule for item pairs
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nAssociation Rules :\n", rules[['antecedents','consequents','support','confidence','lift']])
'''
Association Rules :
                  antecedents               consequents  ...  confidence      lift
0                (Gladiator)             (Sixth Sense)  ...    0.714286  1.190476
1              (Sixth Sense)               (Gladiator)  ...    0.833333  1.190476
2                  (Patriot)             (Sixth Sense)  ...    0.666667  1.111111
3              (Sixth Sense)                 (Patriot)  ...    0.666667  1.111111
4                  (Patriot)               (Gladiator)  ...    1.000000  1.428571
5                (Gladiator)                 (Patriot)  ...    0.857143  1.428571
6       (Gladiator, Patriot)             (Sixth Sense)  ...    0.666667  1.111111
7     (Patriot, Sixth Sense)               (Gladiator)  ...    1.000000  1.428571
8   (Gladiator, Sixth Sense)                 (Patriot)  ...    0.800000  1.333333
9                  (Patriot)  (Gladiator, Sixth Sense)  ...    0.666667  1.333333
10               (Gladiator)    (Patriot, Sixth Sense)  ...    0.571429  1.428571
11             (Sixth Sense)      (Gladiator, Patriot)  ...    0.666667  1.111111

Most of the conbination contains mainly 3 movies (Patriot,Sixth Sense,Gladiator)
for above pairwise association rule as well as for individual assoiciation rules
for these movies. 

General Interpretation:
Viewers who enjoy Gladiator, Patriot, and Sixth Sense tend to have strong 
cross-genre preferences—especially between action, war, and psychological 
thrillers—making them ideal targets for bundled recommendations and multi-genre
marketing strategies.

Benefits/Impact of the Solution:
1. Finds which movies are often liked together  
2. Helps create movie bundles for promotions  
3. Suggests films based on viewer taste  
4. Improves targeting in ads and campaigns  
5. Boosts viewer satisfaction with better recommendations  
6. Increases chances of repeat viewing and loyalty  
7. Helps plan themed movie packs across genres  
8. Uses data to guide marketing and content strategy  
'''
