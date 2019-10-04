# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:24:54 2019

@author: Ashwin
"""
#Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

sns.set(style='white', context='notebook', palette='deep')

#Load Data
def main():
    dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    data = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    test_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
    
    
    prof_train=dataset[['Profession']]
    prof_test =test_data[['Profession']]
    
    #dataset.to_excel('after_processing.xlsx')
    
    
    df = pd.concat([prof_train,prof_test])
    df = df.reset_index(drop=True)
    
    df_gpby = df.groupby(list(df.columns))
    
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    
    refactor=df.reindex(idx)
    
    for i in idx:
        if i<111800:
            dataset=dataset.drop(i)
    
    # Check for Null Data
    dataset.isnull().sum()
    # Replace All Null Data in NaN
    dataset = dataset.fillna(np.nan)
    test_data =  test_data.fillna(np.nan)
    # Get data types
    dataset.dtypes
    # Peek at data
    dataset.head(4)
    dataset = dataset[dataset.Income<2000000]
    dataset = dataset[dataset.Income>0]
    print(dataset)
    
    #Analyze Data
    
    # Identify Numeric features
    numeric_features = ['Instance','Size of City','Wears Glasses','Body Height [cm]','Year of Record','Age','Income']
    # Identify Categorical features
    cat_features = ['Hair Color','University Degree','Profession', 'Country', 'Gender']
    
    #dataset["Hair Color"] = dataset["Hair Color"].astype(str)
    #dataset["University Degree"] = dataset["University Degree"].astype(str)
    #dataset["Profession"] = dataset["Profession"].astype(str)
    #dataset["Country"] = dataset["Country"].astype(str)
    #dataset["Gender"] = dataset["Gender"].astype(str)
    
    #g = sns.heatmap(dataset[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
    
    # Fill Missing Category Entries
    dataset["Hair Color"] = dataset["Hair Color"].fillna("Unknown")
    dataset["University Degree"] = dataset["University Degree"].fillna("Unknown")
    dataset["Profession"] = dataset["Profession"].fillna("Unknown")
    dataset["Country"] = dataset["Country"].fillna("Unknown")
    dataset["Gender"] = dataset["Gender"].fillna("unknown")
    
    dataset["Year of Record"] = dataset["Year of Record"].fillna(dataset["Year of Record"].mean())
    dataset["Year of Record"] = np.round_(dataset["Year of Record"])
    
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset["Age"] = np.round_(dataset["Age"])
    
    # Reformat Column We Are Predicting
    dataset["Hair Color"] = dataset["Hair Color"].replace(to_replace ="0", value ="Unknown")
    dataset["University Degree"] = dataset["University Degree"].replace(to_replace ="0", value ="Unknown") 
    dataset["Gender"] = dataset["Gender"].replace(to_replace ="0", value ="unknown")
    
     # Fill Missing Category Entries in test data
    test_data["Hair Color"] = test_data["Hair Color"].fillna("Unknown")
    test_data["University Degree"] = test_data["University Degree"].fillna("Unknown")
    test_data["Profession"] = test_data["Profession"].fillna("Unknown")
    test_data["Country"] = test_data["Country"].fillna("Unknown")
    test_data["Gender"] = test_data["Gender"].fillna("unknown")
    
    test_data["Year of Record"] = test_data["Year of Record"].fillna(test_data["Year of Record"].mean())
    test_data["Year of Record"] = np.round_(test_data["Year of Record"])
    
    test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
    test_data["Age"] = np.round_(test_data["Age"])
    
    # Reformat Column We Are Predicting
    dataset["Hair Color"] = dataset["Hair Color"].replace(to_replace ="0", value ="Unknown")
    dataset["University Degree"] = dataset["University Degree"].replace(to_replace ="0", value ="Unknown") 
    dataset["Gender"] = dataset["Gender"].replace(to_replace ="0", value ="unknown")
    
    # Reformat Column We Are Predicting for test data
    test_data["Hair Color"] = test_data["Hair Color"].replace(to_replace ="0", value ="Unknown")
    test_data["University Degree"] = test_data["University Degree"].replace(to_replace ="0", value ="Unknown") 
    test_data["Gender"] = test_data["Gender"].replace(to_replace ="0", value ="unknown") 
    
    #g = sns.barplot(x="Hair Color",y="Income",data=dataset)
    #g = sns.barplot(x="University Degree",y="Income",data=dataset)
    #g = sns.barplot(x="Profession",y="Income",data=dataset)
    #g = sns.barplot(x="Country",y="Income",data=dataset)
    #g = sns.barplot(x="Gender",y="Income",data=dataset)
    
    # Confirm All Missing Data is Handled
    dataset.isnull().sum()
    
    #Feature Engineering
    dataset["Hair Color"].value_counts()
    dataset["University Degree"].value_counts()
    dataset["Profession"].value_counts()
    dataset["Country"].value_counts()
    dataset["Gender"].value_counts()
    
    
    
            
    
    #choose = 'Body Height [cm]','Year of Record','Age'
    #choose = 'University Degree','Gender'
    #dataset.drop(labels=['Instance','Size of City','Wears Glasses','Hair Color','Profession', 'Country'], axis = 1, inplace = True)
    
    # copy dataset for training
    dataset_train = dataset[['Year of Record','Age','Gender','Profession','University Degree','Body Height [cm]','Income']].copy()
    dataset_train.isnull().sum()
    dataset_train.dtypes
    
    from sklearn.preprocessing import LabelBinarizer
    lb_style = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    dataset_train = dataset_train.join(pd.DataFrame(lb_style.fit_transform(dataset_train["Gender"]),columns=lb_style.classes_, index=dataset_train.index))
    dataset_train = dataset_train.join(pd.DataFrame(lb_style.fit_transform(dataset_train["University Degree"]),columns=lb_style.classes_, index=dataset_train.index))
    dummy=pd.get_dummies(dataset_train["Profession"],drop_first=True)
    prof=dummy.as_matrix()
    #dataset_train = dataset_train.join(dummy)
    
    dataset_train.dtypes
    
    # copy dataset for test data
    dataset_test = test_data[['Year of Record','Age','Gender','Profession','University Degree','Body Height [cm]']].copy()
    dataset_test.isnull().sum()
    dataset_test.to_excel('y_data.xlsx')
    
    lb_test_style = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    dataset_test =  dataset_test.join(pd.DataFrame(lb_style.fit_transform(dataset_test["Gender"]),columns=lb_style.classes_, index=dataset_test.index))
    dataset_test = dataset_test.join(pd.DataFrame(lb_style.fit_transform(dataset_test["University Degree"]),columns=lb_style.classes_, index=dataset_test.index))
    dummy_test= pd.get_dummies(dataset_test["Profession"],drop_first=True)
    test_prof = dummy_test.as_matrix()
    dataset_test.dtypes
    
    corrmat = dataset_train.corr() 
  
    f, ax = plt.subplots(figsize =(9, 8)) 
    sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
    
    array = dataset_train.values
    X = np.concatenate((array[:,[0,1,5,7,8,9,10,11,12,13,14,15]],prof),axis=1)
    Y = array[:,6]
    
    test_array = dataset_test.values
    zeros=np.zeros([73230,8)
    X_test = np.concatenate((test_array[:,[0,1,5,6,7,8,9,10,11,12,13,14]],test_prof,zeros),axis=1)
    
   
    #Modeling
    
    validation_size = 0.20
    seed = 11
    num_folds = 10
    scoring = 'accuracy'
    #X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,test_size=validation_size,random_state=seed)
    X_train = X
    Y_train = Y
    # Params for Random Forest
    num_trees = 100
    max_features = 3
    reg = LinearRegression().fit(X_train, Y_train)
    b=reg.coef_
    print(b)
    reg.predict(X_train)
    aa=np.mean((Y_train-reg.predict(X_train)) ** 2)
    a=reg.predict(X_test)
    print(aa)
    np.savetxt('out.csv',a)
    
    #Spot Check 5 Algorithms (LR, LDA, KNN, CART, GNB, SVM)
    #models = []
    #models.append(('LR', LogisticRegression()))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    #models.append(('CART', DecisionTreeClassifier()))
    #models.append(('NB', GaussianNB()))
    #models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
    #models.append(('SVM', SVC()))
    # evalutate each model in turn
    '''
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)
        
    fig = plt.figure()
    fig.suptitle('Algorith Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    #Algorithm Tuning
    '''
'''
Commented Out to Reduce Script Time - Took 20 Minutes to run.
best n_estimator = 250
best max_feature = 5
# Tune Random Forest
n_estimators = np.array([50,100,150,200,250])
max_features = np.array([1,2,3,4,5])
param_grid = dict(n_estimators=n_estimators,max_features=max_features)
model = RandomForestClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''

#Finalizing the Model

# a) Predictions on validation dataset - KNN
'''
random_forest = RandomForestClassifier(n_estimators=250,max_features=5)
random_forest.fit(X_train, Y_train)
predictions = random_forest.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''



