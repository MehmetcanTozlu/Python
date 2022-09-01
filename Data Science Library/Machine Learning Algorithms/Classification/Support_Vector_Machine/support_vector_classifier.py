import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Ulke.csv')

x = df.iloc[:, 1:4].values
y = df.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
