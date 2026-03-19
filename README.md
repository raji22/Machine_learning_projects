Data Preprocessing

Overview:
  Data preprocessing is an important step in this machine learning project. Raw data usually contains missing values, noise, and inconsistencies.
  So, we clean and transform the data before feeding it into the model.
  
Steps Involved:
1. Data Cleaning  
      Removed missing/null values using appropriate methods (mean/median/imputation)
      Handled duplicate records
      Fixed inconsistent data formats

2. Handling Missing Values
      Numerical columns → filled using mean/median
      Categorical columns → filled using mode

3. Encoding Categorical Data
      Converted categorical features into numerical format using:
      Label Encoding
      One-Hot Encoding

4. Feature Scaling
      Applied scaling techniques to normalize data:
        Standardization (Z-score)
        Min-Max Scaling
   
Libraries Used:
    pandas
    numpy
    sklearn
