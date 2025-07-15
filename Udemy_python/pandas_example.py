import pandas as pd

df1 = pd.DataFrame([[2,4,6], [10,20,30]], columns=['Price', 'Age', 'Value'])
df2 = pd.DataFrame([{"Name": "John", "Surname": "Johns"}, {"Name": "Jack"}])

print(df1.Price.mean())
print(df1.Price.max())