import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('C:/Users/mehmet/Desktop/Data Science/Data Sets/bisiklet_fiyatlari.xlsx')

x = df.iloc[:, 1:3].values
y = df.iloc[:, 0:1].values

plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='k-means++', random_state=0)
y_pred = km.fit_predict(x)
print(km.cluster_centers_)
plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1], c='red')
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1], c='blue')
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1], c='green')
plt.show()

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++')
    km.fit_predict(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.show()

from sklearn.cluster import KMeans
km2 = KMeans(n_clusters=6, init='k-means++')
km2.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=km2.labels_, cmap='turbo')
plt.show()
