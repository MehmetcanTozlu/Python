import numpy as np
import pandas as pd

df = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)

# Data Preprocessing
# Stop Word, anlamsiz kelimeleri cikaralim(it, that, the gibi).
import nltk
nltk.download('stopwords') # ingilizce stopwords olan kelimeleri indirdik

# kelimelerin koklerini bulmak icin gerekli kutuphanemiz
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords

# Veri Setimizde ki noktalama isaretlerini kaldiralim.
import re # Regular Expression Kutuphanemizi ekledik.

derlem = []

# kelimeleri kucuk harfe ceviren,
# her birini liste elemani olarak ayarlayan,
# stop words olup olmadigini kontrol eden,
# en sonda da tekrar string haline getiren bir filtre yazdik.
for i in range(716):
    # kucuk ve buyuk harf icermeyenleri filtrele ve yerine bosluk karakterini koy.
    comment = re.sub('[^a-zA-Z]', ' ', df['Review'][i]) 
    comment = comment.lower()
    comment = comment.split()
    comment = [ ps.stem(word) for word in comment if not word in set(stopwords.words('english')) ] # govdeyi bul
    comment = ' '.join(comment)
    derlem.append(comment)


# Feature Extraction / BOW(Bag Of Words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000) # en fazla kullanilan 1000 tane kelimeyi al
X = cv.fit_transform(derlem).toarray() # independent variable
y = df.iloc[:, -1] # dependent variable


# Machine Learning Algorithms
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

X_train = X_train.reshape(1, -1)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

