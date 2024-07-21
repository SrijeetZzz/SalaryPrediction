import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data=pd.read_csv("/content/Salary_Data.csv")
print(data)

sns.heatmap(data.isnull())

minsal=data['Salary'].mean()
data['Salary'].fillna(minsal,inplace=True)
sns.heatmap(data.isnull())

x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,1].values
print(y)

plt.scatter(x,y,color="red")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.plot(x,model.predict(x),color="blue")

inp=[[2.5]]
pred_sal=model.predict(inp)
print(pred_sal)
plt.scatter(inp,pred_sal,color="green")


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)