import os
import pandas
from sklearn.preprocessing import StandardScaler


scale = StandardScaler()
df = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/cars2.csv")
X = df[["Weight", "Volume"]]
scaledX = scale.fit_transform(X)
print(scaledX)
