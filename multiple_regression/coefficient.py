import os
import pandas
from sklearn import linear_model


df = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/cars.csv")
X = df[["Weight", "Volume"]]
y = df["CO2"]
regression = linear_model.LinearRegression()
regression.fit(X, y)
print(regression.coef_)
