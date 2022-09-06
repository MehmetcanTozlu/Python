import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('musteriler.csv')

x = df.iloc[:, 3:].values

# Kutuphanemiz
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred = ac.fit_predict(x)

plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1])
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1])
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1])
plt.scatter(x[y_pred == 3,0], x[y_pred == 3,1])
plt.show()

plt.scatter(x[:,0:1], x[:, 1:2], c=ac.labels_, cmap='jet')
plt.show()

# Dendrogram icin gereken kutuphanemiz
from scipy.cluster.hierarchy import dendrogram, linkage
dendrogram = dendrogram(linkage(x,method='ward'))
plt.show()
