#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

#Importing the dataset
tips_data = pd.read_csv("tips.csv")

#Displaying first 5 rows of the dataset
print(tips_data.head())

#preprocessing the dataset
print(tips_data.shape)
print(tips_data.nunique())
print(tips_data.columns)
print(tips_data.dtypes)
print(tips_data.info())
print("@@@@@@@@@@@@@@@@@")
print(tips_data.describe())
print("@@@@@@@@@@@@@@@@@")

print(tips_data['time'].unique())
print("@@@@@@@@@@@@@@@@@")
print(tips_data['day'].value_counts())
print("@@@@@@@@@@@@@@@@@")
print(tips_data['day'].value_counts().sum())
print("@@@@@@@@@@@@@@@@@")
print(tips_data.isnull().sum())

plt.hist(tips_data['total_bill'], bins=20)
plt.title("Total Bill Distribution")
plt.show()

plt.scatter(tips_data['total_bill'], tips_data['tip'])
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.show()

sns.boxplot(x='sex', y='tip', data=tips_data)
plt.title("Tip Distribution by Gender")
plt.show()

sns.barplot(x='day', y='tip', data=tips_data)
plt.title("Average Tip by Day")
plt.show()

sns.countplot(x='day', data=tips_data)
plt.title("Number of Customers per Day")
plt.show()

corr = tips_data.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
# sns.heatmap(tips_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(tips_data)
plt.show()

#Encoding the Data
le = LabelEncoder()
tips_data["sex"] = le.fit_transform(tips_data["sex"])
tips_data["smoker"]=le.fit_transform(tips_data["smoker"])
tips_data["day"] = le.fit_transform(tips_data["day"])
tips_data["time"]=le.fit_transform(tips_data["time"])

print(tips_data.head())


x=tips_data.drop(["tip"],axis=1)
y=tips_data["tip"]
scaler = StandardScaler()
x=scaler.fit_transform(x)

# Split dataset into train and test sets
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=42)

# Initialize Random Forest Regressor
regressor = RandomForestRegressor()

# Define hyperparameter grid
param_grid = {

    'criterion':['squared_error','absolute_error','friedman_mse','poisson'],
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=regressor,
    param_grid=param_grid,
    cv=5,                 # 5-fold cross validation
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
#
y_pred = best_model.predict(x_test)
print(y_pred)
#
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))