# SC1015 FCCA Group2 
Data Science Mini-Project
Members: Hu Han, Zhu Xinuo, Lai Yi
Main Theme: Obesity Risk Prediction

## 0.Note: 
(1) The main program is a jupyter notebook "Obesity.ipynb"
(2) The classification module is put in a class "Classification" in a separate .py file "Classification.py".
(3) Some lines may run very slowly, so please wait patiently.


## 1.Motivation:
We aim to better inform the public about the factors related to obesity using data science techniques.
Our goal is to build a model to predict obesity levels based on variables other than heights and weights, which can be useful for health screening organizations to predict the obesity risk and provide advice.

## 2.Problem Definition: 
How are the different variables related to obesity risk prediction?

## 3.Data:
### (1)Source:
“Obesity or CVD risk (Classify/Regressor/Cluster)”
https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster

### (2)Introduction:
The data consist of the estimation of obesity levels in people from the countries of Mexico, Peru and Colombia, with ages between 14 and 61 and diverse eating habits and physical condition, data was collected using a web platform with a survey where anonymous users answered each question, then the information was processed obtaining 17 attributes and 2111 records. 

### (3)Variable Names:
#### Attributes related with eating habits:
Frequent consumption of high caloric food (FAVC), 
Frequency of consumption of vegetables (FCVC), 
Number of main meals (NCP), 
Consumption of food between meals (CAEC), 
Consumption of water daily (CH20), and 
Consumption of alcohol (CALC). 

#### Attributes related with the physical condition: 
Calories consumption monitoring (SCC), 
Physical activity frequency (FAF), 
Time using technology devices (TUE), 
Transportation used (MTRANS)

#### Other variables:
Age, Gender, family_fistory_with_overweight, SMOKE

#### Variable types:
Numerical Variables: Age, FCVC, NCP, CH2O, FAF, TUE
Categorical Variables: Gender, family_history_with_overweight, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS

### (4)Target variable definition:
NObeyesdad: classified based on BMI Range
Insufficient_Weight: (0, 18.5)
Normal_Weight: [18.5, 25.0)
Overweight_Level_I: [25.0, 27.5)
Overweight_Level_II: [27.5, 30.0)
Obesity_Type_I: [30.0, 35.0)
Obesity_Type_II: [35.0, 40.0)
Obesity_Type_III: [40.0, +∞)

# 4.Exploratory Analysis/Visualisation
(1)Draw diagrams of the predictors and the target.
(2)Draw diagrams of the relationship between the predictors and the target.

# 5.Cleaning & Preprocess 
(1)Remove missing or duplicated values.
(2)Encode the categorical variables to numerical, including the target.

# 6.Machine Learning Process
(1)Single Decision Trees
(2)Multivariate Decision Tree
(3)Random Forest
(4)Other models


