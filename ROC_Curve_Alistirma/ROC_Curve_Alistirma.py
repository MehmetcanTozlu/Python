import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv('C:/users/mehmet/desktop/data science/data sets/ulke.csv')

x = df.iloc[:, 1:4].values
y = df.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini')
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Decision Tree: \n', cm)

# K-Nearest-Neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('KNN: \n', cm)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Logistic Regression:\n', cm)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gbayes = GaussianNB()
gbayes.fit(X_train, y_train)
y_pred = gbayes.predict(X_test)
y_proba = gbayes.predict_proba(X_test) # tahmin olasiliklarini getir, % kac erkek % kac kadin gibi
cm = confusion_matrix(y_test, y_pred)
print('Naive Bayes: \n', cm)

# Roc Curve Bayes
from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:, 0], pos_label='e')
# False Positive Rate, True Positive Rate ve Threshold dondurur 
print(fpr)
print(tpr)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Random Forest: \n', cm)

# SVM
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('SVM: \n', cm)
