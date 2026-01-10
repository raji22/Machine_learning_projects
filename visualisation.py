import pandas as pd

df = pd.read_csv("world_top_restaurants_dataset.csv")
print(df.head())

columns = df.columns
print(columns)
star_category_unique = df['Star_Category'].unique()
print(star_category_unique)