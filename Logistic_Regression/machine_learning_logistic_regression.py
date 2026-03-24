#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#Read the file
df = pd.read_csv("weather_forecast_data.csv")
print(df.head())
#Dimension of the Dataframe

print(df.shape)

#Description about the dataset
print(df.describe())
#Detail information about the dataset
print(df.info())

#Finding Null values And sum of the null values
print(df.isnull())
print(df.isnull().sum())

#checking duplicate value#checking duplicate value
print(df.duplicated().sum())

#Finding column's value count
total_rain_possible= df['Rain'].value_counts()
print(total_rain_possible)

#Convert the categorical data into numerical data(non categorical)
from sklearn.preprocessing import LabelEncoder
rain_label = LabelEncoder()
df['Rain'] = rain_label.fit_transform(df['Rain'])
print(df.head())

#Visualisation (Temperature vs Rain)
plt.scatter(df["Temperature"],df["Rain"])
plt.xlabel("Temperature")
plt.ylabel("Rain (0=No, 1=Yes)")
plt.title("Temperature vs Rain")
plt.show()

# #Correlation between rating and numeric features
import seaborn as sns
plt.figure(figsize=(10,6))
num_cols = ['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']
corr = df[num_cols].corr()
# sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1, square=True)
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0
)
plt.title('Correlation Matrix — numeric feature relationships')
plt.show()

#Temperature Distribution
sns.histplot(df['Temperature'], kde=True)
plt.title("Temperature Distribution")
plt.show()

#Model-Logistic Regression

#Features and Target
x= df[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']]
y= df['Rain']

#Train Test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# standardize the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
print("@@@@@@@")
print(x_test.shape)

model =LogisticRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(y_predict)

print("Accuracy:", accuracy_score(y_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))
