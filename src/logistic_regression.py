'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here
def run_logistic_regression(df_arrests):
    # define features and target
    features = ['current_charge_felony', 'num_fel_arrests_last_year']
    target = 'y'

    # split data
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests[target],
        random_state=42
    )

    X_train = df_arrests_train[features]
    y_train = df_arrests_train[target]
    X_test = df_arrests_test[features]

    # 2. training with GridSearchCV with 3 values of C, 5-fold CV
    param_grid = {'C': [0.01, 1.0, 1.0]}
    lr_model = lr(solver='liblinear', max_iter=1000, random_state=42)
    gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy')
    gs_cv.fit(X_train, y_train)

    # Report best hyperparameter
    best_c = gs_cv.best_params_['C']
    print(f"\nQ: What was the optimal value for C?")
    print(f"A: {best_c}")

    if best_c == min(param_grid['C']):
        reg_level = "most regularization (smallest C)"
    elif best_c == max(param_grid['C']):
        reg_level = "least regularization (largest C)"
    else:
        reg_level = "in the middle"

    print(f"\nQ: Did it have the most or least regularization? Or in the middle?")
    print(f"A: {reg_level}")

    # predict probabilities on test set
    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_lr'] = gs_cv.predict_proba(X_test)[:, 1]

    # save for 4 and 5
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

    return df_arrests_train, df_arrests_test



