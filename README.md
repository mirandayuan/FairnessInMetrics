# Fairness In Machine Learning

## Data

LOS mimic dataset

protected features: age, gender and race

  
## Preprocess Data：
### standardize

    run 'LOS_features.py'
    //convert string value to integers.
    //there are many missing values in this dataset:
    //while using xgboost method to train the model, we do not need to care about the missing data;
    //while using other methods, we could fill the missing data with 0 or median value:
    //fill with 0: uncomment line 113 and line 115;
    //fill with median: uncomment line 113 and line 114.
    
### calculate the correlation with target and protected features

    run 'correlation.py'
    //since there are both nominal variables and continuous variables in the dataset, we use r square to do the evaluation

### split the dataset to training dataset and test dataset

    run 'split.py'
    
### Resample to balance the size of each group: 

    run 'balance.py'
    //balance the distribution of different groups of training dataset(but keep the ratio of ‘yes’ to ‘no’ of each group)
    //'num' in line 13 is the size of training set after balancing
        
    
## Train the Model:

  choose one from the following four methods
  
  A. Logistic Regression: run 'logistic.py'
  
  B. Random Forest: run 'random.py'
  
  C. Gradient Boosting Decision Tree: run 'gbdt.py'
 
  D. xgboost: run 'xgb.py'
  
  sort the features by the importance of features (based on the model trained by xgboost method):
  
    run 'imp.py'
 
 
## Analyze the results:
 
### Calculate metrics and Plot curves(ROC curve and precision&recall&f1score for different thresholds curve):

    run 'analyze_LOS.py' 
    //for age groups: uncomment the code from line 70 to line 86
    //for gender groups: uncomment the code from line 91 to line 105
    //for race groups: uncomment the code from line 112 to line 130
    
  
### Find the optimal threshold by computing average error
  
    run 'findthreshold.py'
    //for age groups: uncomment the code from line 60 to line 64, and line 50
    //for gender groups: uncomment the code from line 67 to line 70, and line 51
    //for race groups: uncomment the code from line 73 to line 77, and line 52 
  
