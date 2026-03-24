'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10, model_name="Model"):
    """
    A calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(prob_true, bin_means, marker='o', label=model_name)
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Plot, {model_name}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'data/calibration_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

# draw plot calibration
def run_calibration_plots(df_arrests_test):
    """
    draw calibration plots and comparison metrics.
    
    parameters: 
        df_arrests_test : pd.DataFrame
    """
    
    y = df_arrests_test['y']
    lr = df_arrests_test['pred_lr']
    dt = df_arrests_test['pred_dt']

    # calibration plots
    calibration_plot(y, lr, n_bins=5, model_name="Logistic Regression")
    calibration_plot(y, dt, n_bins=5, model_name="Decision Tree")

    # which is more calibrated?
    frac_lr, mean_lr = calibration_curve(y, lr, n_bins=5)
    frac_dt, mean_dt = calibration_curve(y, dt, n_bins=5)
    err_lr = np.mean(np.abs(frac_lr - mean_lr))
    err_dt = np.mean(np.abs(frac_dt - mean_dt))

    if err_lr < err_dt:
        better = "Logistic Regression"
    else:
        better = "Decision Tree"
    
    print(f"\nQ: Which model is more calibrated?")
    print(f"A: {better} (lower calibration error).")

    # extra credit 
    ppv_lr = df_arrests_test.nlargest(50, 'pred_lr')['y'].mean()
    ppv_dt = df_arrests_test.nlargest(50, 'pred_dt')['y'].mean()
    print(f"\nPPV for Logistic Regression (top 50): {ppv_lr:.4f}")
    print(f"PPV for Decision Tree (top 50):       {ppv_dt:.4f}")

    auc_lr = roc_auc_score(y, lr)
    auc_dt = roc_auc_score(y, dt)
    print(f"\nAUC for Logistic Regression: {auc_lr:.4f}")
    print(f"AUC for Decision Tree:       {auc_dt:.4f}")

    ppv_winner = "Logistic Regression" if ppv_lr > ppv_dt else "Decision Tree"
    auc_winner = "Logistic Regression" if auc_lr > auc_dt else "Decision Tree"
    print(f"\nQ: Do both metrics agree that one model is more accurate?")
    if ppv_winner == auc_winner:
        print(f"A: Yes, both agree {ppv_winner} is more accurate.")
    else:
        print(f"A: No, PPV favors {ppv_winner}, AUC favors {auc_winner}.")
