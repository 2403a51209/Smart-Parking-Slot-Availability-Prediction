import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("parking_dataset.csv")

X = df.drop("occupancy", axis=1)
X["timestamp"] = pd.to_datetime(X["timestamp"]).astype(int) // 10**9
y = df["occupancy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("Models trained successfully")
