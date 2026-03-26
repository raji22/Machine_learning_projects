#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Load Dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")
print(df.head())
print(df.info())

#Handle Missing Values
df = df.drop(['salary'], axis=1)   # Not needed for prediction

#Encode Categorical Columns
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

#Define Features & Target
X = df.drop(['status', 'sl_no'], axis=1)
y = df['status']

#Apply ANOVA Feature Selection
#Select top 5 important features:

anova = SelectKBest(score_func=f_classif, k=5)
X_selected = anova.fit_transform(X, y)

selected_features = X.columns[anova.get_support()]
print("Selected Features:", selected_features)

#View F-Scores & p-values
scores = pd.DataFrame({
    "Feature": X.columns,
    "F_score": anova.scores_,
    "p_value": anova.pvalues_
})

print(scores.sort_values(by="F_score", ascending=False))

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)

#Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Compare Without Feature Selection
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_train_full, y_train)

y_pred_full = model_full.predict(X_test_full)

print("Accuracy without feature selection:",
      accuracy_score(y_test, y_pred_full))