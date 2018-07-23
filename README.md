# FairnessInMetrics

I. Data:

A. Pima Indians Diabetes Dataset(source: https://www.kaggle.com/uciml/pima-indians-diabetes-database): 

  a) 'diabetes.csv'
  
B. Throid Disease Dataset from UCI(source: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease):

  a) training: 'allbp.data.txt'
  
  b) test: 'allbp.test.txt'
  
  
  
  
II. Preprocess Data：

  A. For diabetes dataset:
  
    run 'diabetes_features.py'
    
  B. For thyroid disease dataset:
  
    run 'thy_features.py'
  
  
    
    
III. Train the Model:

  choose one from the following three methods
  
  A. Logistic Regression: run 'logistic.py'
  
  B. Random Forest: run 'random.py'
  
  C. Gradient Boosting Decision Tree: run 'gbdt.py'
 
 
 
 
 IV. Analyze the results:
 
  Calculate metrics and Plot curves(ROC curve and precision&recall&f1score for different thresholds curve):
  
  A. For diabetes dataset:
  
    run 'analyze_dia.py'
    
  B. For thyroid disease dataset:
  
    run 'analyze_thy.py'
  
  
