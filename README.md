<p align="center">
  <img src="docs/readme/banner.svg" alt="Model Comparison. Classification, calibration, and the tradeoffs between them." width="100%">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-453C70?style=flat-square">
  <a href="https://github.com/gi11ikin/problem-set-2"><img alt="View upstream repository" src="https://img.shields.io/badge/source-upstream-64748b?style=flat-square"></a>
</p>

<p align="center"><a href="src/main.py">Entry point</a> &nbsp; · &nbsp; <a href="src/calibration_plot.py">Calibration code</a></p>

## Overview

A course exercise comparing logistic regression and decision trees on arrest-event data. The workflow covers data preparation, train/test separation, parameter selection, predicted probabilities, and calibration diagnostics.

This repository is a personal fork of [the original course repository](https://github.com/gi11ikin/problem-set-2). The original instructions are retained below.

## At a glance

| Area | What to look for |
| --- | --- |
| **Prepare** | Retrieve the course datasets and construct model features and outcomes. |
| **Compare** | Tune logistic regression and decision-tree parameters with cross-validation. |
| **Evaluate** | Inspect calibration, ROC AUC, and positive predictive value among the highest-ranked predictions. |

## Start here

From the repository root, in an activated environment:

```sh
pip install -r requirements.txt
python src/main.py
```

## Scope

The source datasets are loaded from external course links. This is a modeling exercise, not a validated tool for decisions about individuals. End-to-end execution and current link availability were not rechecked during this documentation refresh.

---

<details>
<summary><strong>Original course instructions</strong></summary>

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
