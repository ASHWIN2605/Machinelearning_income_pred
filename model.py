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
    #Removing Outliers and negative values from Income of TrainData
    if IsTrainData == True:
        dataset = dataset[dataset.Income<2600000]
        dataset = dataset[dataset.Income>0]
    
    #Analyze Data
    # Fill Missing Category Entries with Unknown so we get a separtae column of unknown in dataset
    dataset["Hair Color"] = dataset["Hair Color"].fillna("Unknown")
    dataset["University Degree"] = dataset["University Degree"].fillna("Unknown")
    dataset["Profession"] = dataset["Profession"].fillna("Unknown_Profession")
    dataset["Country"] = dataset["Country"].fillna("Unknown_Country")
    dataset["Gender"] = dataset["Gender"].fillna("unknown")
    
    #Fill missing numeric entries with the mean of the overall entries
    dataset["Year of Record"] = dataset["Year of Record"].fillna(dataset["Year of Record"].mean())
    dataset["Year of Record"] = np.round_(dataset["Year of Record"])
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset["Age"] = np.round_(dataset["Age"])
    
    # Reformat Column so as to replace zero values with unknown
    dataset["Hair Color"] = dataset["Hair Color"].replace(to_replace ="0", value ="Unknown")
    dataset["Profession"] = dataset["Profession"].replace(to_replace ="0", value ="Unknown_Profession")
    dataset["Country"] = dataset["Country"].replace(to_replace ="0", value ="Unknown_Country")
    
    
    # Confirm All Missing Data is Handled
    dataset.isnull().sum()
    
    #return the dataset
    return dataset

    
#Function to maintain consistency between train and test data input features
#Inputs:Train and test data
#output :Modified dataframe having consistent input datas between Train and Test Data  
def Handle_consistency(dataset,test_data):
    
    #Get the categorical features whcih needs to part of Training Model
    categorical_features = ['Country','Profession','University Degree']
    
    #Copying only these categorical data in separate dataframe.
    dataset_train = dataset[categorical_features]
    dataset_test = test_data[categorical_features]
    
    #Adding separate column in both dataframe at last and giving different values to it
    dataset_train["train"]=1
    dataset_test["train"]=0
    
    ##Concatenating both the train and test data so if get dummies is performed after this,we aill get unique values of both Train and Test Data
    combined=pd.concat([dataset_train,dataset_test])
    
    #Changing all the Profession and Country list to Uppercase,so as to remove duplicates while performing get dummies
    combined['Profession']=combined['Profession'].apply(lambda x: x.upper())
    combined['Country']=combined['Country'].apply(lambda x: x.upper())
    
    
    #Perform get dummies in the concatenated dataframe for the above mentioned categorical data and storing it in separate dattaframe
    prof_data=pd.get_dummies(combined[categorical_features],drop_first=False,columns=categorical_features)
    
    #Concatenating the obtained encoded columns in the actual combined dataframe
    combined=pd.concat((combined,prof_data),axis=1)
    
    #Splitting the Train and test dataframe using the values of the Train column added.
    train_prof = combined[combined['train'] == 1]
    test_prof = combined[combined['train'] == 0]
    
    #Droping the original categorical columns and the added train column from the modified train dataframe
    mod_train =train_prof.drop(['Country','Profession','University Degree','train'],axis=1)
    
    #Concatenating the required numerical features to the modifed train dataframe and this corresponds to our X_Train
    X= pd.concat((dataset[['Year of Record','Age','Body Height [cm]']],mod_train),axis=1)
    
    #Getting the Income of train data as Y and taking it's log so as to maintain easy processing.
    Y =dataset[['Income']]
    Y=np.log(Y)

    #Droping the original categorical columns and the added train column from the modified test dataframe
    mod_test =test_prof.drop(['Country','Profession','University Degree','train'],axis=1)
    
     #Concatenating the required numerical features to the modifed train dataframe and this corresponds to our X_Test
    X_test= pd.concat((test_data[['Year of Record','Age','Body Height [cm]']],mod_test),axis=1)
    
    #Return the values
    return X,Y,X_test
 



    
    
#Train the model to predict the values
#Inputs X_Train Y_Train X_Validate,Y_Validate and X_test
#Output Predicted value
def Train_Model(X_Train,Y_Train,X_Test):
    #fitting Bayseian Regression to the training dataset
    regressor = BayesianRidge()
    fitResult = regressor.fit(X_Train, Y_Train)
    
    #predicting the model for X_test
    b=fitResult.predict(X_Test)
    
    #Performing exponential function on the predicted values and exporting the output values to an out excel file
    b=np.exp(b)
    np.savetxt('out.csv',b)
         
           
#Run all the function calls from here
def run():
    #Load Data
    dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    test_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
    
    #Pre-Process the Data
    Pre_process_Train=Pre_ProcessingData(dataset,True)
    Pre_Process_Test=Pre_ProcessingData(test_data,False)

    #Handle consistency between train and test data
    X_Encoded,Y_Encoded,X_Test_Encoded=Handle_consistency(Pre_process_Train, Pre_Process_Test)
    
    #Model_data
    Train_Model(X_Encoded,Y_Encoded,X_Test_Encoded)

#main call   
def main():
    #function call to execute the run
    run()
if __name__ == '__main__':
    main()
    




