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

sns.set(style='white', context='notebook', palette='deep')

#Library to import LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

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
    dataset["Country"] = dataset["Country"].fillna("Unknown")
    dataset["Gender"] = dataset["Gender"].fillna("unknown")
    
    dataset["Year of Record"] = dataset["Year of Record"].fillna(dataset["Year of Record"].mean())
    dataset["Year of Record"] = np.round_(dataset["Year of Record"])
    
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset["Age"] = np.round_(dataset["Age"])
    
    # Reformat Column so as to replace zero values with unknown
    dataset["Hair Color"] = dataset["Hair Color"].replace(to_replace ="0", value ="Unknown")
    dataset["University Degree"] = dataset["University Degree"].replace(to_replace ="0", value ="Unknown") 
    dataset["Profession"] = dataset["Profession"].replace(to_replace ="0", value ="Unknown_Profession")
    dataset["Gender"] = dataset["Gender"].replace(to_replace ="0", value ="unknown")
    
    
    # Confirm All Missing Data is Handled
    dataset.isnull().sum()
    
    #Feature Engineering
    dataset["Hair Color"].value_counts()
    dataset["University Degree"].value_counts()
    dataset["Profession"].value_counts()
    dataset["Country"].value_counts()
    dataset["Gender"].value_counts()
    
    return dataset

#Function to create dummies for the categorical data and covert it into numerical format
#input :dataset and bool to determine training dataset or not
#output :modified dataset
def Handle_Dummies(dataset):
    
    # copy dataset for training.Taken only cerain parameters for consideration
    dataset_train = dataset[['Year of Record','Age','Gender','Profession','University Degree','Body Height [cm]','Income']].copy()
    dataset_train.isnull().sum()
    dataset_train.dtypes

    #Converting the column data of categorical data into separate columns 
    
    dataset_train = dataset_train.join(pd.DataFrame(lb_style.fit_transform(dataset_train["Gender"]),columns=lb_style.classes_, index=dataset_train.index))
    dataset_train = dataset_train.join(pd.DataFrame(lb_style.fit_transform(dataset_train["University Degree"]),columns=lb_style.classes_, index=dataset_train.index))
    return dataset_train
    
 #Function to maintain consistency between train and test data input features
#Inputs:Train and test data
#output :Modified dataframe having consistent input datas   
def Handle_consistency(dataset,test_data,IsTrainData,testing):
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
        
        if IsTrainData == True:
            X =dataset.drop(['Gender','Profession','University Degree','Income'],axis=1)
            Y =dataset[['Income']]
            return X,Y
        else:
            X=test_data.drop(['Gender','Profession','University Degree','Income'],axis=1)
            return X
    
#Train the model to predict the values
#Inputs X_Train and Y_Train
#Output Predicted value
def Train_Model(X_Train,Y_Train,X_Test):
    #Modeling by fitting in Linear reg
    reg = LinearRegression().fit(X_Train, Y_Train)
    b=reg.coef_
    print(b)
    reg.predict(X_Train)
    RMSE = np.mean((Y_Train-reg.predict(X_Train)) ** 2)
    Predict = reg.predict(X_Test)
    np.savetxt('out.csv',Predict)
  
    
#Run all the function calls from here
def run():
    #Load Data
    dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    test_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
    
    #Pre-Process the Data
    Pre_process_Train=Pre_ProcessingData(dataset,True)
    Pre_Process_Test=Pre_ProcessingData(test_data,False)
    
     #to handle dummy values
    X_Trian_Transform=Handle_Dummies(Pre_process_Train)
    X_Test_Transform=Handle_Dummies(Pre_Process_Test)
    
    
    #handle consistency between train and test data
    X_Train,Y_Train=Handle_consistency(X_Trian_Transform,X_Test_Transform,True,False)
    X_Test = Handle_consistency(X_Trian_Transform,X_Test_Transform,False,False)
    
    
    #Train_data
    Train_Model(X_Train,Y_Train,X_Test)

#main call   
def main():
    #function call to execute the run
    run()
if __name__ == '__main__':
    main()
    




