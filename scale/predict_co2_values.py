import os
import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


scale = StandardScaler()
df = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/cars2.csv")
X = df[["Weight", "Volume"]]
y = df["CO2"]
scaledX = scale.fit_transform(X)
regression = linear_model.LinearRegression()
regression.fit(scaledX, y)
scaled = scale.transform([[2300, 1.3]])
predictedCO2 = regression.predict([scaled[0]])
print(predictedCO2)
