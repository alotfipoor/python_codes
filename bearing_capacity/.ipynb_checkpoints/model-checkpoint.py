# Common imports
import numpy as np
import os
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# loading the data
bearing = pd.read_csv("bearing_capacity_data.csv")
#bearing_fit = bearing["RMR","γ(kN/m3)","D(m)","B(m)","qu(MPa)", "S/B", "ϕ(o)"]
bearing_fit = bearing.drop('qult(MPa)', axis=1)
bearing_label = bearing['qult(MPa)']
#print("bearing_fit")

# details about the dataframe
#bearing.head()
#bearing.info()
#bearing["RMR"].value_counts()
#bearing.describe()

# making histograms for data
#bearing.hist(bins=50, figsize=(20,15))
#plt.show()

# preparing the test data with simple spliting
#from sklearn.model_selection import train_test_split
#train_set, test_set = train_test_split(bearing, test_size=0.2, random_state=42)

# visualize the data to gain insights
#bearing.plot(kind="scatter", y="RMR", x="qult(MPa)")
#bearing.plot(kind="scatter", y="qu(MPa)", x="qult(MPa)")
#bearing.plot(kind="scatter", y="S/B", x="qult(MPa)")
#bearing.plot(kind="scatter", y="γ(kN/m3)", x="qult(MPa)")
#plt.legend()

# preprocess the categorical input feature
#from sklearn.preprocessing import OneHotEncoder
#cat_encoder = OneHotEncoder()
#bearing_cat = bearing[['RMR']]
#bearing_cat_encoded = cat_encoder.fit_transform(bearing_cat)
#bearing_cat_encoded.toarray()

# the ML model
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import cross_val_score

#forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

#scores = cross_val_score(forest_reg, bearing_fit, bearing_label,
#                         scoring="neg_mean_squared_error", cv=100)

#print("Scores:", scores)
#print("Mean:", scores.mean())
#print("Standard deviation:", scores.std())


# grid search
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(bearing_fit, bearing_label)

#The best hyperparameter combination found:
grid_search.best_params_
grid_search.best_estimator_
