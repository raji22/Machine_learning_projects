import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# df = pd.read_csv("rounded_hours_student_scores.csv")
df = pd.read_csv("score_updated.csv")
# df = pd.read_csv("score.csv")
print(df)
print(df.tail())
print(df.shape)

print(df.columns)

print(df.info())
print(df.describe())
print(df.isnull().sum().sum())

#Linear Regression Model
x = df[["Hours"]]
y = df["Scores"]

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=42)
result = LinearRegression()
result.fit(x_train,y_train)
# result.fit(x,y)
print(x_train.shape)
print("@@@@@@@@@@@@@@@@@@@@@@")
print(x_test.shape)

# final_result = pd.DataFrame({"Hours":["9.3","9.5","5.8","8.8"]})
# predicted_value = result.predict(final_result)
# final_result["predicted_value"] = predicted_value
# print(final_result)

final_predict = result.predict(x_test)
print(final_predict)

#Visualise
plt.scatter(x_test, y_test, color='b', label='Actual Data')
plt.plot(x_test, final_predict,color='r', label='Regression Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Linear Regression')
plt.show()

#Acurracy checking
from sklearn.metrics import r2_score
final_check = r2_score(y_test,final_predict)
print(final_check)