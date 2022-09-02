import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

# Iris veri setini analiz edelim

# DataFrame'imiz
df = pd.read_csv('Iris.csv')

x = df.iloc[:, 1:5].values # Bagimsiz Degiskenlerimiz - Index Column'unu almiyoruz
y = df.iloc[:, -1:].values # Bagimli Degiskenlerimiz

# Verilerimizi Train ve Test olarak ayiralim
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Bazi degerlerimiz 0.2 bazilari 5 oldugundan Verilerimizi Olcekleyelim Modelimiz daha dogru calissin
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

# Modellerimizi yazalim ve karsilastiralim

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
cm_log = confusion_matrix(y_test, y_pred_log)
print('\nLogistic Regression\n', cm_log)


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print('\nKNN\n', cm_knn)

# SVM
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print('\nSVC\n', cm_svc)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
print('\nDecision Tree\n', cm_dtree)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
cm_rfc = confusion_matrix(y_test, y_pred_rfc)
print('\nRandom Forest\n', cm_rfc)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gau_bayes = GaussianNB()
gau_bayes.fit(X_train, y_train)
y_pred_bayes = gau_bayes.predict(X_test)
cm_bayes = confusion_matrix(y_test, y_pred_bayes)
print('\nNaive Bayes\n', cm_bayes)











