# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:24:54 2019

@author: Ashwin
"""
#Load Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn import metrics


#Preprocssing function.
#Inputs Dataframe,IsTraindata
#Output Cleaned dataFrame of X_Train and Y_Train
def Pre_ProcessingData(dataset,IsTrainData):
    
    # Check for Null Data and removing it
    dataset.isnull().sum()
    # Replace All Null Data in NaN
    dataset = dataset.fillna(np.nan)
    # Get data types
    dataset.dtypes
    # Peek at data
    dataset.head(4)
    #Removing Outliers and negative values from Income
    if IsTrainData == True:
        dataset = dataset[dataset.Income<5000000]
        dataset = dataset[dataset.Income>0]
    
    #Analyze Data
    # Identify Numeric features
    numeric_features = ['Instance','Size of City','Wears Glasses','Body Height [cm]','Year of Record','Age','Income']
    # Identify Categorical features
    cat_features = ['Hair Color','University Degree','Profession', 'Country', 'Gender']
    
    # Fill Missing Category Entries with Unknown so we get a separtae column of unknown in dataset
    dataset["Hair Color"] = dataset["Hair Color"].fillna("Unknown")
    dataset["University Degree"] = dataset["University Degree"].fillna("Unknown")
    dataset["Profession"] = dataset["Profession"].fillna("Unknown_Profession")
    dataset["Country"] = dataset["Country"].fillna("Unknown_Country")
    dataset["Gender"] = dataset["Gender"].fillna("unknown")
    
    dataset["Year of Record"] = dataset["Year of Record"].fillna(dataset["Year of Record"].mean())
    dataset["Year of Record"] = np.round_(dataset["Year of Record"])
    
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset["Age"] = np.round_(dataset["Age"])
    
    # Reformat Column so as to replace zero values with unknown
    dataset["Hair Color"] = dataset["Hair Color"].replace(to_replace ="0", value ="Unknown")
    #dataset["University Degree"] = dataset["University Degree"].replace(to_replace ="0", value ="Unknown") 
    dataset["Profession"] = dataset["Profession"].replace(to_replace ="0", value ="Unknown_Profession")
    #dataset["Gender"] = dataset["Gender"].replace(to_replace ="0", value ="unknown")
    dataset["Country"] = dataset["Country"].replace(to_replace ="0", value ="Unknown_Country")
    
    
    # Confirm All Missing Data is Handled
    dataset.isnull().sum()
    
    #Feature Engineering
    dataset["Hair Color"].value_counts()
    dataset["University Degree"].value_counts()
    dataset["Profession"].value_counts()
    dataset["Country"].value_counts()
    dataset["Gender"].value_counts()

    return dataset

     
    
#Function to maintain consistency between train and test data input features
#Inputs:Train and test data
#output :Modified dataframe having consistent input datas   
def Handle_consistency(dataset,test_data,testing):
    if testing == True:
        dataset = dataset.join(pd.DataFrame(lb_style.fit_transform(dataset["Profession"]),columns=lb_style.classes_, index=dataset.index))
        #Add a new column -> Other Profession
        dummyData = [0] * len(dataset['Profession'])
        dataset['Other Profession'] = dummyData
    
        refProfessionList = list(set(test_data['Profession'].unique()) - set(dataset['Profession'].unique()))
        test_data['Profession'] = test_data['Profession'].replace(refProfessionList,'Other Profession')
    
        test_data = test_data.join(pd.DataFrame(lb_style.fit_transform(test_data["Profession"]),columns=lb_style.classes_, index=test_data.index))
    
        #Add a new columns to match the training frame
        dummy_test_Data = [0] * len(test_data['Profession'])
        neededProfessionList = list(set(dataset['Profession'].unique()) - set(test_data['Profession'].unique()))

        for prof in neededProfessionList:
            test_data[prof] = dummy_test_Data
        
        X =dataset.drop(['Gender','Profession','University Degree','Income'],axis=1)
        Y =dataset[['Income']]
        X_test=test_data.drop(['Gender','Profession','University Degree','Income'],axis=1)
        return X,Y,X_test
    else:
        categorical_features = ['Country','Profession','University Degree']
        dataset_train = dataset[categorical_features]
        #print(dataset_train)
        dataset_test = test_data[categorical_features]
        dataset_train["train"]=1
        dataset_test["train"]=0
        combined=pd.concat([dataset_train,dataset_test])
        prof_data=pd.get_dummies(combined[categorical_features],drop_first=True,columns=categorical_features)
        combined=pd.concat((combined,prof_data),axis=1)
        train_prof = combined[combined['train'] == 1]
        test_prof = combined[combined['train'] == 0]
        mod_train =train_prof.drop(['Country','Profession','University Degree','train'],axis=1)
        X= pd.concat((dataset[['Year of Record','Age','Body Height [cm]']],mod_train),axis=1)
        print(X)
        Y =dataset[['Income']]
        mod_test =test_prof.drop(['Country','Profession','University Degree','train'],axis=1)
        X_test= pd.concat((test_data[['Year of Record','Age','Body Height [cm]']],mod_test),axis=1)
        return X,Y,X_test



#Test_train Split Function
def Test_Train_split(X_Train,Y_Train,X_Test):
    X_train, X_test, y_train, y_test = train_test_split(X_Train, Y_Train, test_size=0.2)
    #print (X_train.shape, y_train.shape)
    #print (X_test.shape, y_test.shape)
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_Test = scaler.transform(X_Test)
    #X_test=scaler.transform(X_test)
    return X_train,y_train,X_test,y_test,X_Test
    
    
    
#Train the model to predict the values
#Inputs X_Train and Y_Train
#Output Predicted value
def Train_Model(X_Train,Y_Train,X_Validate,Y_Validate,X_Test,Is_Linear_Reg):
    if Is_Linear_Reg:
        
        #Modeling by fitting in Linear reg
        reg = LinearRegression().fit(X_Train, Y_Train)
        b=reg.coef_
        #print(b)
        reg.predict(X_Validate)
        RMSE = np.sqrt(mean_squared_error(Y_Validate,(reg.predict(X_Validate))))
        print(RMSE)
        Predict = reg.predict(X_Test)
        np.savetxt('out.csv',Predict)
    else:
         #clf_ = SGDRegressor(alpha=0.0001, average=False, early_stopping=False,
       #epsilon=0.1, eta0=0.01, fit_intercept=True, l1_ratio=0.15,
       #learning_rate='invscaling', loss='squared_loss', max_iter=1000,
       #n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
       #shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
       #warm_start=False)
         #clf_.fit(X_Train, Y_Train)
         #clf_.predict(X_Validate)
         #RMSE = np.sqrt(mean_squared_error(Y_Validate,(clf_.predict(X_Validate))))
         #print(RMSE)
         #b=clf_.predict(X_Test)
         #np.savetxt('out.csv',b)
         #fitting SVR to the dataset
         #from sklearn.svm import SVR
         #svr_reg = SVR(kernel = 'rbf', gamma = 'scale', epsilon= 0.2)
         #svr_reg.fit(X_Train, Y_Train)
         #b=svr_reg.predict(X_Test)
         #np.savetxt('out.csv',b)
         regressor = BayesianRidge()
         fitResult = regressor.fit(X_Train, Y_Train)
         YPredTest = regressor.predict(X_Validate)
         RMSE=np.sqrt(metrics.mean_squared_error(Y_Validate, YPredTest))
         print(RMSE)
         b=regressor.predict(X_Test)
         np.savetxt('out.csv',b)
         
         
       
      
    
#Run all the function calls from here
def run():
    #Load Data
    dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    test_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
    
    #Pre-Process the Data
    Pre_process_Train=Pre_ProcessingData(dataset,True)
    Pre_Process_Test=Pre_ProcessingData(test_data,False)

    #handle consistency between train and test data
    X_Encoded,Y_Encoded,X_Test_Encoded=Handle_consistency(Pre_process_Train, Pre_Process_Test, False)
    
    #Train_trest Split
    X_Train,Y_Train,X_Validate,Y_Validate,X_Test=Test_Train_split(X_Encoded,Y_Encoded,X_Test_Encoded)
    
    
    #Train_data
    Train_Model(X_Train,Y_Train,X_Validate,Y_Validate,X_Test,False)

#main call   
def main():
    #function call to execute the run
    run()
if __name__ == '__main__':
    main()
    




