# Regression and Decision Trees

![Python](docs/readme/badges/python-3776AB.svg)
![pandas](docs/readme/badges/pandas-150458.svg)
![scikit-learn](docs/readme/badges/scikitlearn-F7931E.svg)

A coursework comparison of logistic regression and decision trees using arrest records. The code constructs a one year rearrest outcome, derives charge and prior arrest features, searches model settings with cross validation, and compares calibration and ranking metrics.

## My contribution

I implemented preprocessing, model fitting, and evaluation modules on top of the [course starter](https://github.com/gi11ikin/problem-set-2). The original instructions remain below for context.

## Explore the implementation

- [Preprocessing](src/preprocessing.py) defines the outcome and time based features.
- [Logistic regression](src/logistic_regression.py) and [decision trees](src/decision_tree.py) use grid search with five fold cross validation.
- [Calibration](src/calibration_plot.py) compares probabilities and includes AUC and precision among the top 50 ranked records.

## What the comparison is designed to show

The logistic regression model adjusts regularization through `C`; the decision tree varies maximum depth. Both use five fold cross validation. Calibration asks whether predicted probabilities agree with observed frequencies, while AUC and precision among the top 50 records ask different ranking questions.

Keeping those questions separate is central to the exercise. A better ranking score does not by itself establish reliable probabilities, fair treatment across groups, or suitability for real decisions.

## Running and current limitations

Install `requirements.txt` in a fresh Python environment. The actual entry point is `python src/main.py`, run from the repository root. The original assignment's reference to a root `main.py` does not match this checkout.

The ETL module uses `pandas.read_csv` for remote URLs named `.feather`; those inputs and their format need resolving before treating this as reproducible end to end. No verified model scores are claimed here. These are classroom analyses of recorded arrests, not validated estimates of criminal behavior or tools for decisions about individuals.

<details>
<summary>Original course assignment</summary>

PROBLEM SET #2: REGRESSION AND DECISION TREES

Instructions: 
- Clone the Problem Set code package from GitHub: 
- You will use arrest data for each part of this assignment. The links are found in src/etl.py in the code package. 
- - Remember to spend some time to get to know the data before you start on pre-processing and analysis
- Each of the .py files in `/src` contain instructions for the exepected code you are to write


Before you start:
- Make sure to setup a virtual environment as discussed in the Course Tech Setup lecture. Here's a short article as an additional resource: 
https://www.freecodecamp.org/news/python-requirementstxt-explained/
- Don't forget to set your requirements.txt file using pip freeze > requirements.txt 

When you're done:
- Commit and push this code package to your GitHub account
- - A good practice is to use simple commit comments and  commit code after you've finished each feature. Commit and push often

Submission: 
You will submit the GitHub URL for this repo in ELMS.

Grading: 
We will look to make sure you've output the correct CSV files. We will only run main.py, so make sure that you stucture this correctly. Credit will be given for adhering to the course's Code Standards and Data Standards, using GitHub correctly, and producing the correct output, among other considerations.

</details>

![Decorative project banner: Regression. Decision trees. Questions about prediction.](docs/readme/footer.svg)

---

## Author

**Kenneth Yeaher**  
Master of Information Management  
University of Maryland, College Park  
[![LinkedIn: Kenneth Yeaher](https://img.shields.io/badge/LinkedIn-Kenneth_Yeaher-0A66C2?style=flat)](https://www.linkedin.com/in/kennethyeaher/)

`Python` · `Classification` · `Calibration`
