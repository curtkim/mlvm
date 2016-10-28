import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


#use all digits
mnist = fetch_mldata("MNIST original")
X_train, y_train = mnist.data[:70000] / 255., mnist.target[:70000]

# print X_train.shape #(70000,784)

#X_train, y_train = shuffle(X_train, y_train)
#X_train, y_train = X_train[:1000], y_train[:1000]  # lets subsample a bit for a first impression

pca = RandomizedPCA(n_components=2)
fig, plot = plt.subplots()
fig.set_size_inches(10, 10)
plt.prism()

X_transformed = pca.fit_transform(X_train)
plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train)
plot.set_xticks(())
plot.set_yticks(())

#plt.tight_layout()
plt.savefig("mnist_pca.png")
