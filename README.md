# FairnessInMetrics

Data:

Pima Indians Diabetes Dataset(source: https://www.kaggle.com/uciml/pima-indians-diabetes-database): 

  'diabetes.csv'
  
Throid Disease Dataset from UCI(source: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease):

  training: 'allbp.data.txt'
  
  test: 'allbp.test.txt'
  
  
  
  
Preprocess Data：

  For diabetes dataset:
  
    run 'diabetes_features.py'
    
  For thyroid disease dataset:
  
    run 'thy_features.py'
  
  
    
    
Train the Model:

  choose one from the following three methods
  
  1. Logistic Regression: run 'logistic.py'
  
  2. Random Forest: run 'random.py'
  
  3. Gradient Boosting Decision Tree: run 'gbdt.py'
 
 
 
 
 Analyze the results:
 
  Calculate metrics and Plot curves(ROC curve and precision&recall&f1score for different thresholds curve):
  
  For diabetes dataset:
  
    run 'analyze_dia.py'
    
  For thyroid disease dataset:
  
    run 'analyze_thy.py'
  
  
