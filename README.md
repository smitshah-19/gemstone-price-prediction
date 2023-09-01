# Gemstone Price Prediction - Smit Shah
### Introduction
The goal of the project is to predict the ***price*** of a given diamond (Regression Analysis). There are in total ***10*** independent variables and are listed as below: 
- `id`:  Unique identifier of each diamond.
- `Carat`:  Carat weight of the cubic zirconia.
- `Cut`:   Describe the cut quality of the cubic zirconia. Quality is increasing order Fair, Good, Very Good,  Premium, Ideal.
- `Color`:  Colour of the cubic zirconia.With D being the best and J the worst.
- `Clarity`:  Cubic zirconia Clarity refers to the absence of the Inclusions and Blemishes. (In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3.
- `Depth`:  The Height of a cubic zirconia, measured from the Culet to the table, divided by its average Girdle Diameter.
- `Table`:  The Width of the cubic zirconia's Table expressed as a Percentage of its Average Diameter.
- `X`:  Length of the cubic zirconia in mm.
- `Y`:  Width of the cubic zirconia in mm.
- `Z`:  Height of the cubic zirconia in mm.

Target variable:
- `Price`:  The Price of the cubic zirconia.

Source of Dataset: [https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

More details: [American Gem Society](https://www.americangemsociety.org/ags-diamond-grading-system/)

### Azure Deployment Link
Auzre Link: [https://1gemstonepriceprediction.azurewebsites.net/](https://1gemstonepriceprediction.azurewebsites.net/)

# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables IterativeImputer is applied, then ordinal encoding performed. Lastly, after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was catboost regressor.
    * After this hyperparameter tuning is performed on catboost and knn model.
    * A final VotingRegressor is created which will combine prediction of catboost, xgboost and knn models.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the gemstone prices inside a Web Application.
  
# Exploratory Data Analysis Notebook

Link : [EDA Notebook](./notebook/1_EDA_Gemstone_Price_Prediction.ipynb)

# Model Training Approach Notebook

Link : [Model Training Notebook](./notebook/2_Model_Training_Gemstone.ipynb)
