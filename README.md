# FairnessInMetrics

I. Data:

A. Pima Indians Diabetes Dataset(source: https://www.kaggle.com/uciml/pima-indians-diabetes-database): 

B. LOS mimic dataset
  
  
  
  
  
II. Preprocess Data：

  A. For diabetes dataset:
  
    run 'diabetes_features.py'
    
  B. For LOS dataset:
  
    run 'LOS_features.py'

    > Resample to balane the size of each group: 
      
      run 'balance.py'

  
  
    
    
III. Train the Model:

  choose one from the following four methods
  
  A. Logistic Regression: run 'logistic.py'
  
  B. Random Forest: run 'random.py'
  
  C. Gradient Boosting Decision Tree: run 'gbdt.py'
 
  D. xgboost: run 'xgb.py'
 
 
 IV. Analyze the results:
 
  A. Calculate metrics and Plot curves(ROC curve and precision&recall&f1score for different thresholds curve):
  
    run 'analyze_dia.py'  ###for diabete dataset
    
    or
    
    run 'analyze_LOS.py'  ###for LOS dataset
    
    
  
  B. Find the optimal threshold by computing average error
  
    run 'findthreshold.py'

  
  
