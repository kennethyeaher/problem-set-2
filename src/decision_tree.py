'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

# my code 

def run_decision_tree(df_arrests_train, df_arrests_test):
    
    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    # 3 values of max_depth, 5-fold CV
    param_grid_dt = {'max_depth': [2, 5, 10]}
    dt_model = DTC(random_state=42)
    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy')
    gs_cv_dt.fit(df_arrests_train[features], df_arrests_train['y'])

    # print statements
    best_depth = gs_cv_dt.best_params_['max_depth']
    print(f"\nQ: What was the optimal value for max_depth?")
    print(f"A: {best_depth}")

    if best_depth == 2:
        print("\nQ: Did it have the most or least regularization? Or in the middle?")
        print("A: Most regularization (shallowest tree)")
    elif best_depth == 10:
        print("\nQ: Did it have the most or least regularization? Or in the middle?")
        print("A: Least regularization (deepest tree)")
    else:
        print("\nQ: Did it have the most or least regularization? Or in the middle?")
        print("A: In the middle")

    # predict probabilities on test set
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_dt'] = gs_cv_dt.predict_proba(df_arrests_test[features])[:, 1]

    # save to data folder
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

    return df_arrests_test



