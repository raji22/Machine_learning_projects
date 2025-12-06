#Import Libraries
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

#Read the file
df = pd.read_csv("fruit_classification_dataset.csv")
print(df)

#Dimension of the Dataframe
print(df.shape)

#Size of the Dataframe
print(df.size)

#Detail information about the dataset
data_info = df.info()
print(data_info)

#Description about the dataset
data_describe  = df.describe()
print(data_describe)

#Datatypes of each of the column
col_datatypes = df.dtypes
print(col_datatypes)

#Finding Null values
find_null = df.isnull()
print(find_null)

find_null_sum = df.isnull().sum()
print(find_null_sum)

#Finding column's value count
total_color = df['color'].value_counts()
print(total_color)

total_taste = df['taste'].value_counts()
print(total_taste)

total_fruit_name = df['fruit_name'].value_counts()
print(total_fruit_name)

total_fruit_numbers = df['fruit_name'].value_counts().sum()
print(total_fruit_numbers)

total_shape = df['shape'].value_counts()
print(total_shape)

#Displays the unique name of the category(column)

unique_fruit = df["fruit_name"].unique()
print(unique_fruit)

unique_fruit_count = df["fruit_name"].nunique()
print(unique_fruit_count)

unique_shape = df["shape"].unique()
print(unique_shape)

unique_taste = df["taste"].unique()
print(unique_taste)

unique_color = df["color"].unique()
print(unique_color)

#checking duplicate value
df.duplicated().sum()
df.drop_duplicates(inplace = True)
df.duplicated().sum()
print(df)

#Convert the categorical data into numerical data(non categorical)
from sklearn.preprocessing import LabelEncoder

leobj = LabelEncoder()
df['shape'] = leobj.fit_transform(df['shape'])

leobj = LabelEncoder()
df['color'] = leobj.fit_transform(df['color'])

leobj = LabelEncoder()
df['fruit_name'] = leobj.fit_transform(df['fruit_name'])

print(df)

ordinalobj = OrdinalEncoder()
df[['taste']] = ordinalobj.fit_transform(df[['taste']])
print(df)

print(df.columns)

x=df[['size (cm)', 'shape', 'weight (g)', 'avg_price (₹)', 'color', 'taste']]
y=df['fruit_name']

#Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(x)

scaled_df = pd.DataFrame(scaled, columns=x.columns)
print("************************")
print(scaled_df.head())
print(scaled_df.tail())

# scaler = StandardScaler()
# scaled = scaler.fit_transform(df)
#
# scaled_df = pd.DataFrame(scaled, columns=df.columns)
# print("************************")
# print(scaled_df.head())
# print(scaled_df.tail())