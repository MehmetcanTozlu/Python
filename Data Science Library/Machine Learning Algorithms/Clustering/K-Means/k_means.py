import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('musteriler.csv')

x = df.iloc[:, 3:].values # hacim ve maas

# Kutuphanemizi yukluyoruz
from sklearn.cluster import KMeans
# 3 cluster olustur ve k-means++'i kullan
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(x) # x ile train edelim

# 2 li degerlerin birincisi;
# 1. Cluster'de ki Hacimin Center Point'i / Maas'in Center Point'i
# 2. Cluster'de ki Hacimin Center Point'i / Maas'in Center Point'i
# 3. Cluster'de ki Hacimin Center Point'i / Maas'in Center Point'i
print(kmeans.cluster_centers_)

total = [] # WCSS Degerlerini Toplayalim

# Farkli Cluster Sayisi ile En Optimum Cluster Sayisini Bulmaya Calisalim
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123)
    kmeans.fit(x)
    total.append(kmeans.inertia_) # inertia -> WCSS degerlerimiz

# Elbow Method(Dirsek Metodu)
plt.plot(range(1, 11), total)
