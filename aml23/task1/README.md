# TASK 1: PREDICT THE AGE OF A BRAIN FROM MRI FEATURES

## Solution by Samuel Cestola, Khanh Vu, Mario Markov

**Imputation:** MICE algorithm (IterativeImputer from sklearn).  
**Preprocessing:** We remove features with low variance (variance values that are near 0), and remove highly correlated features based on Pearson's correlation (>0.84).  
**Outlier detection**: We use an unsupervised learning technique that measures datapoints' deviation from its neighbors. The metric is called local outlier factor (LOF).  
**Feature selection**: We estimate the importance of features using random forest and select the top 200.  
**Modeling**: The final model is a stacking model (Ridge) that combines Ridge, SVM, and Gaussian Processes.