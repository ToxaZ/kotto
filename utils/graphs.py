__author__ = 'ffuuugor'
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv').values.copy()

X = df[:, 1:-1]
y = df[:, -1]
colors = "bgrcmykw"
# target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
for c, i in zip(colors, range(1,10)):
    cl = "Class_%d" % i
    print c, cl
    plt.scatter(X_r[y == cl, 0], X_r[y == cl, 1], c=c)
plt.legend()
plt.title('PCA of IRIS dataset')

plt.figure()
for c, i in zip(colors, range(0,9)):
    cl = "Class_%d" % i
    print c,cl
    plt.scatter(X_r2[y == cl, 0], X_r2[y == cl, 1], c=c)
plt.legend()
plt.title('LDA of IRIS dataset')

plt.show()