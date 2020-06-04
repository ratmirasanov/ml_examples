import os
import pandas
from sklearn import linear_model


df = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/cars.csv")
X = df[["Weight", "Volume"]]
y = df["CO2"]
regression = linear_model.LinearRegression()
regression.fit(X, y)
# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
predictedCO2 = regression.predict([[2300, 1300]])
print(predictedCO2)
