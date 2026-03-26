'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    pred_universe_raw, arrest_events_raw = etl.run_etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.run_preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test = logistic_regression.run_logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_test = decision_tree.run_decision_tree(df_arrests_train, df_arrests_test)

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.run_calibration_plots(df_arrests_test)


if __name__ == "__main__":
    main()