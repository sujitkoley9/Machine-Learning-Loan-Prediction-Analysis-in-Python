#----------------- Step 1:  Importing required packages for this problem ------------------------------------- 
   # data analysis and wrangling
    import pandas as pd
    import numpy as np
    import random as rn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt

    # machine learning
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import cross_val_score
    
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    from xgboost.sklearn import XGBRegressor
    from xgboost  import plot_importance
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    
    
#--------- Step 2:  Reading and loading train and test datasets and generate data quality report---------------- 
      
    
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    
    
    
    # loading train and test sets with pandas 
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    #append two train  and test dataframe
    full  = train_df.append(test_df,ignore_index=True)
    
    # Print the columns of dataframe
    print(full.columns.values)
    
    # Returns first n rows
    full.head(10)
    
    
    # Retrive data type of object and no. of non-null object
    full.info()
    
    # Retrive details of integer and float data type 
    full.describe()
    
    # To get  details of the categorical types
    full.describe(include=['O'])

   

  #--------------------Prepare data quality report---------------------
  # To get count of no. of NULL for each data type columns = full.columns.values
    columns = full.columns.values
    data_types = pd.DataFrame(full.dtypes, columns=['data types'])
    
    missing_data_counts = pd.DataFrame(full.isnull().sum(),
                            columns=['Missing Values'])
    
    present_data_counts = pd.DataFrame(full.count(), columns=['Present Values'])
    
    UniqueValues = pd.DataFrame(full.nunique(), columns=['Unique Values'])
    
    MinimumValues = pd.DataFrame(columns=['Minimum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) | (full[c].dtypes == 'int64'):
            MinimumValues.loc[c]=full[c].min()
       else:
            MinimumValues.loc[c]=0
 
    MaximumValues = pd.DataFrame(columns=['Maximum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) |(full[c].dtypes == 'int64'):
            MaximumValues.loc[c]=full[c].max()
       else:
            MaximumValues.loc[c]=0
    
    data_quality_report=data_types.join(missing_data_counts).join(present_data_counts).join(UniqueValues).join(MinimumValues).join(MaximumValues)
    data_quality_report.to_csv('Loan_data_quality.csv', index=True)



   
   
   
#-----------------------------------Step 3: Missing value treatment-----------------------------------   
   # Treatment for Gender
   full['Gender'].fillna('Missing', inplace=True) 
   
   
    # Treatment for Married  
   full['Married'].fillna('Missing', inplace=True)
  
     # Treatment for Self_Employed
  
   full['Self_Employed'].fillna('Missing', inplace=True)
   
     # Treatment for Dependents
  
   full['Dependents'].fillna('Missing', inplace=True)
   
   # Treatment for Credit_History
  
   full['Credit_History'].fillna(1, inplace=True)
   
    # Treatment for Loan_Amount_Term  
   full['Loan_Amount_Term'].fillna(full['Loan_Amount_Term'].mean(), inplace=True)
   
    # Treatment for LoanAmount  
   full['LoanAmount'].fillna(full['LoanAmount'].mean(), inplace=True)
        
 
  
#----------------------------------Step 4: Outlier Treatment -----------------------------------------  

    #  outlier treatment using BoxPlot for CoapplicantIncome,ApplicantIncome,
    # LoanAmount and Loan_Amount_Term

     BoxPlot=boxplot(full['CoapplicantIncome'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['CoapplicantIncome'].isin(outlier),'CoapplicantIncome']=full['CoapplicantIncome'].mean()
     
     #ApplicantIncome
     BoxPlot=boxplot(full['ApplicantIncome'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['ApplicantIncome'].isin(outlier),'ApplicantIncome']=full['ApplicantIncome'].mean()

     #LoanAmount
     BoxPlot=boxplot(full['LoanAmount'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['LoanAmount'].isin(outlier),'LoanAmount']=full['LoanAmount'].mean()
    

#-------------------------------Step 5:Exploration analysis of data--------------------------------------

   # Percent of male/female for Loan Status
    full['Loan_Status'] =np.where(full['Loan_Status'] =='Y', 1,0)
   
    Full_Analysis=full[0:613][['Gender','Loan_Status']].groupby('Gender',as_index=False).mean().sort_values(by='Loan_Status', ascending=False)
    Full_Analysis['Loan_Status']=Full_Analysis['Loan_Status']*100
    g= sns.factorplot(x='Gender', 
                      y='Loan_Status',
                   data=Full_Analysis, 
                   hue='Gender',  # Color by Sex
                   kind='bar') # barplot
                    
    # Percent of Credit history for  Loan Status
    Full_Analysis=full[0:613][['Credit_History', 'Loan_Status']].groupby(['Credit_History'], as_index=False).mean().sort_values(by='Loan_Status', ascending=True)
   
    Full_Analysis['Loan_Status']=Full_Analysis['Loan_Status']*100
   
    g= sns.factorplot( x='Credit_History',
                       y='Loan_Status',
                       data=Full_Analysis, 
                       hue='Credit_History',  
                       kind='bar') # barplot

    # Percent of Dependents for  Loan Status
       Full_Analysis= full[0:613][['Dependents','Loan_Status']].groupby('Dependents',as_index=False).mean().sort_values(by='Loan_Status', ascending=False)
       Full_Analysis['Loan_Status']=Full_Analysis['Loan_Status']*100 
       g= sns.factorplot(x='Dependents',
                         y='Loan_Status',
                         data=Full_Analysis, 
                         hue='Dependents',  
                         kind='bar') # barplot
       
   # Percent of Married for  Loan Status
       Full_Analysis= full[0:613][['Married','Loan_Status']].groupby('Married',as_index=False).mean().sort_values(by='Loan_Status', ascending=False)
       Full_Analysis['Loan_Status']=Full_Analysis['Loan_Status']*100 
       g= sns.factorplot(x='Married',
                         y='Loan_Status',
                         data=Full_Analysis, 
                         hue='Married',  
                         kind='bar') # barplot
       
    # Percent of Property_Area for  Loan Status
       Full_Analysis= full[0:613][['Property_Area','Loan_Status']].groupby('Property_Area',as_index=False).mean().sort_values(by='Loan_Status', ascending=False)
       Full_Analysis['Loan_Status']=Full_Analysis['Loan_Status']*100 
       g= sns.factorplot(x='Property_Area',
                         y='Loan_Status',
                         data=Full_Analysis, 
                         hue='Property_Area',  
                         kind='bar') # barplot  
       
    # Percent of Property_Area for  Loan Status
       Full_Analysis= full[0:613][['Education','Loan_Status']].groupby('Education',as_index=False).mean().sort_values(by='Loan_Status', ascending=False)
       Full_Analysis['Loan_Status']=Full_Analysis['Loan_Status']*100 
       g= sns.factorplot(x='Education',
                         y='Loan_Status',
                         data=Full_Analysis, 
                         hue='Education', 
                         kind='bar') # barplot      
       
       
       
 


#------------------------------------Step 6:Feature Engineering--------------------------------------
  
   #Creating dummy variable for Credit_History using get_dummies
  
   Credit_History_dummies = pd.get_dummies(full['Credit_History'],prefix='Credit_History')
   Credit_History_dummies=Credit_History_dummies.iloc[:,1:]
   full=full.join(Credit_History_dummies)
   
   
   #Creating dummy variable for Dependents using get_dummies
   Dependents_dummies = pd.get_dummies(full['Dependents'],prefix='Dependents')
   Dependents_dummies=Dependents_dummies.iloc[:,1:]
   full=full.join(Dependents_dummies)
   
    #Creating dummy variable for Dependents using get_dummies
   Gender_dummies = pd.get_dummies(full['Gender'],prefix='Gender')
   Gender_dummies=Gender_dummies.iloc[:,1:]
   full=full.join(Gender_dummies)
   
   
    #Creating dummy variable for Married using get_dummies
   Married_dummies = pd.get_dummies(full['Married'],prefix='Married')
   Married_dummies=Married_dummies.iloc[:,1:]
   full=full.join(Married_dummies)
   
   
     #Creating dummy variable for Self_Employed using get_dummies
   Self_Employed_dummies = pd.get_dummies(full['Self_Employed'],prefix='Self_Employed')
   Self_Employed_dummies=Self_Employed_dummies.iloc[:,1:]
   full=full.join(Self_Employed_dummies)
   
    #Creating dummy variable for Property_Area using get_dummies
   Property_Area_dummies = pd.get_dummies(full['Property_Area'],prefix='Property_Area')
   Property_Area_dummies=Property_Area_dummies.iloc[:,1:]
   full=full.join(Property_Area_dummies)
   
   #Creating dummy variable for Education using get_dummies
   Education_dummies = pd.get_dummies(full['Education'],prefix='Education')
   Education_dummies=Education_dummies.iloc[:,1:]
   full=full.join(Education_dummies)
   
    #Creating dummy variable for Income
   Education_dummies = pd.get_dummies(full['Education'],prefix='Education')
   Education_dummies=Education_dummies.iloc[:,1:]
   full['Total_Income']=full['ApplicantIncome']+full['CoapplicantIncome']
   
    #---------------------------------- Droping unnecessary columns-------------------------------
    full.drop(['Credit_History','Dependents','Education','Gender','Loan_ID','Married',
               'Property_Area','Self_Employed','ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)
    full.columns.values

   
   
   
#----------------------Step 7: Separating train/test dataset and Normalize data--------------------------------   
    train_new=full[0:614]
    test_new=full[614:]
    
    X_train = train_new.drop(['Loan_Status'], axis=1)
    Y_train = train_new["Loan_Status"]
    
    X_test  = test_new.drop(['Loan_Status'], axis=1)
   
    #-----Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
     #--------------------PCA to reduce dimension and remove correlation----------------------------    
     pca = PCA(n_components =16)
     pca.fit_transform(X_train)
     #The amount of variance that each PC explains
     var= pca.explained_variance_ratio_
     #Cumulative Variance explains
     var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
     plt.plot(var1)
     
     # As per analysis, we can skip 4 principal componet, use only 11 components
     
     pca = PCA(n_components =14)
     X_train=pca.fit_transform(X_train)
     X_test=pca.fit_transform(X_test)
     
     
#---------------------------Step 8: Run Algorithm-------------------------------------------------
   #1.Logistic Regression
    
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    logreg_acc = cross_val_score(estimator = logreg, X = X_train, y = Y_train, cv =    10)
    logreg_acc_mean = logreg_acc.mean()
    logreg_std = logreg_acc.std()
    


   #2.Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    decision_tree_acc = cross_val_score(estimator = decision_tree, X = X_train, y = Y_train, cv =    10)
    decision_tree_acc_mean = decision_tree_acc.mean()
    decision_tree_std = decision_tree_acc.std()
    
    # Choose some parameter combinations to try
    parameters = {
                  'criterion': ['entropy', 'gini'],
                  'max_depth': range(2,10), 
                  'min_samples_split': range(2,10),
                  'min_samples_leaf': range(1,10)
                 }

    # Search for best parameters
    grid_obj = GridSearchCV(estimator=decision_tree, 
                                    param_grid= parameters,
                                    scoring = 'accuracy',
                                    cv = 10,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    decision_tree_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    decision_tree_best.fit(X_train, Y_train)
    
    # Calculate accuracy of decisison tree again
    decision_tree_acc = cross_val_score(estimator = decision_tree_best, X = X_train, y = Y_train, cv =    10)
    decision_tree_acc_mean = decision_tree_acc.mean()
    decision_tree_std = decision_tree_acc.std()
    
    #---To Know importanve of variable
    feature_importance = pd.Series(decision_tree_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    
    
    
   #3.Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    random_forest_acc = cross_val_score(estimator = random_forest, X = X_train, y = Y_train, cv =    10)
    random_forest_acc_mean = random_forest_acc.mean()
    random_forest_std = random_forest_acc.std()



    # Choose some parameter combinations to try
    parameters = { 
                 'max_features': ['log2', 'sqrt','auto'], 
                  'criterion': ['entropy', 'gini'],
                  'max_depth': range(2,10), 
                  'min_samples_split': range(2,10),
                  'min_samples_leaf': range(1,10)
                 }
    
    
   
    grid_obj = GridSearchCV(estimator=random_forest, 
                                    param_grid= parameters,
                                    scoring = 'accuracy',
                                    cv = 10,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)
    

    
   

    # Set the clf to the best combination of parameters
    random_forest_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    random_forest_best.fit(X_train, Y_train)
    random_forest_acc = cross_val_score(estimator = random_forest_best, X = X_train, y = Y_train, cv =    10)
    random_forest_acc_mean = random_forest_acc.mean()
    random_forest_std = random_forest_acc.std()
    
    #---To Know importanve of variable
    feature_importance = pd.Series(random_forest_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    

   #4.XGBOOST
    Xgboost = XGBClassifier()
    Xgboost.fit(X_train, Y_train)
    Xgboost_acc = cross_val_score(estimator = Xgboost, X = X_train, y = Y_train, cv =    10)
    Xgboost_acc_mean = Xgboost_acc.mean()
    Xgboost_std = Xgboost_acc.std()



    # Choose some parameter combinations to try
   parameters = {'learning_rate':np.arange(0.1, .5, 0.1),
                  'n_estimators':[1000],
                  'max_depth': range(4,10),
                  'min_child_weight':range(1,5),
                  'reg_lambda':np.arange(0.55, .9, 0.05),
                  'subsample':np.arange(0.1, 1, 0.1),
                  'colsample_bytree':np.arange(0.1, 1, 0.1)
               }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=Xgboost, 
                                  param_distributions = parameters,
                                  scoring = 'accuracy',
                                  cv = 10,n_iter=300,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    Xgboost_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    Xgboost_best.fit(X_train, Y_train)
    Xgboost_acc = cross_val_score(estimator = Xgboost_best, X = X_train, y = Y_train, cv =    10)
    Xgboost_acc_mean = Xgboost_acc.mean()
    Xgboost_std = Xgboost_acc.std()
    
   
    plot_importance(Xgboost_best)
    pyplot.show()
    
   #5.SVM
    SVM_Classifier=SVC()
    SVM_Classifier.fit(X_train, Y_train)
    SVM_Classifier_acc = cross_val_score(estimator = SVM_Classifier, X = X_train, y = Y_train, cv =    10)
    SVM_Classifier_acc_mean = SVM_Classifier_acc.mean()
    SVM_Classifier_std = SVM_Classifier_acc.std()



    # Choose some parameter combinations to try
   parameters = { 'kernel':('linear', 'rbf'),
                  'gamma': [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5],
                  'C': np.arange(1, 10,1)
                 }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=SVM_Classifier, 
                                  param_distributions = parameters,
                                  scoring = 'accuracy',
                                  cv = 3,n_iter=100,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    SVM_Classifier_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    SVM_Classifier_best.fit(X_train, Y_train)
    SVM_Classifier_acc = cross_val_score(estimator = SVM_Classifier_best, X = X_train, y = Y_train, cv =    10)
    SVM_Classifier_acc_mean = SVM_Classifier_acc.mean()
    SVM_Classifier_std = SVM_Classifier_acc.std()
  

   #.6.KNN
    KNN_Classifier=KNeighborsClassifier() 
    KNN_Classifier.fit(X_train, Y_train)
    KNN_Classifier_acc = cross_val_score(estimator = KNN_Classifier, X = X_train, y = Y_train, cv =    10)
    KNN_Classifier_acc_mean = KNN_Classifier_acc.mean()
    KNN_Classifier_std = KNN_Classifier_acc.std()



    # Choose some parameter combinations to try
   parameters = { 'n_neighbors': np.arange(1, 31, 2),
	              'metric': ["euclidean", "cityblock"]
                 }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=KNN_Classifier, 
                                  param_distributions = parameters,
                                  scoring = 'accuracy',
                                  cv = 10,n_iter=30,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    KNN_Classifier_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    KNN_Classifier_best.fit(X_train, Y_train)
    KNN_Classifier_acc = cross_val_score(estimator = KNN_Classifier_best, X = X_train, y = Y_train, cv =    10)
    KNN_Classifier_acc_mean = KNN_Classifier_acc.mean()
    KNN_Classifier_std = KNN_Classifier_acc.std()
    
    
    
  #7.Artificial Neural network    
   # Initialising the ANN
   def build_classifier(optimizer):
        ANN_classifier = Sequential()
        ANN_classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 14))
        ANN_classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
        ANN_classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        ANN_classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        return ANN_classifier
    
    classifier = KerasClassifier(build_fn = build_classifier)
   
    # Choose some parameter combinations to try
    parameters = {'batch_size': [25, 32],
                  'epochs': [10, 20],
                  'optimizer': ['adam', 'rmsprop']
                  }
    
    # Search for best parameters
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
                               scoring = 'accuracy',
                               cv = 3)
    
    grid_search = grid_search.fit(X_train, Y_train)
   
 
    # Set the clf to the best combination of parameters
    ANN_Classifier_best = grid_search.best_estimator_
    
     # Fit the best algorithm to the data. 
    ANN_Classifier_best.fit(X_train, Y_train)
    ANN_Classifier_acc = cross_val_score(estimator = ANN_Classifier_best, X = X_train, y = Y_train, cv =    10)
    ANN_Classifier_acc_mean = ANN_Classifier_acc.mean()
    ANN_Classifier_std = ANN_Classifier_acc.std()
    

 #---------------Step 9:Prediction on test data ----------------------------------------------------------------
     Y_pred1 = logreg.predict(X_test)
     Y_pred2 = decision_tree_best.predict(X_test)
     Y_pred3 = random_forest_best.predict(X_test)
     Y_pred4 = Xgboost_best.predict(X_test)
     Y_pred5 = SVM_Classifier_best.predict(X_test)
     Y_pred6 = KNN_Classifier_best.predict(X_test)
     Y_pred7 = ANN_Classifier_best.predict(X_test)
    
    
     
    submission = pd.DataFrame({
            "Loan_ID": test_df["Loan_ID"],
            "Loan_Status": Y_pred1
        })
    
    submission["Loan_Status"] =np.where( submission["Loan_Status"]==1,'Y','N')
    submission.to_csv('Submission.csv', index=False)





 
 