import idx2numpy
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import gzip

# Task 1: Import data
train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

# Task 2: flatten images
X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)
y_train = train_labels
y_test = test_labels

#Task 3.1 Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Task 3.2: Dimensionality reduction
# PCA examples
pca_50 = PCA(n_components=50).fit(X_train_scaled)
X_train_pca_50 = pca_50.transform(X_train_scaled)
X_test_pca_50 = pca_50.transform(X_test_scaled)

pca_100 = PCA(n_components=100).fit(X_train_scaled)
X_train_pca_100 = pca_100.transform(X_train_scaled)
X_test_pca_100 = pca_100.transform(X_test_scaled)

pca_200 = PCA(n_components=200).fit(X_train_scaled)
X_train_pca_200 = pca_200.transform(X_train_scaled)
X_test_pca_200 = pca_200.transform(X_test_scaled)

# LDA examples 
lda_9 = LDA(n_components=9).fit(X_train_scaled, y_train)
X_train_lda = lda_9.transform(X_train_scaled)
X_test_lda = lda_9.transform(X_test_scaled)


print("PCA 50:", X_train_pca_50.shape, X_test_pca_50.shape)
print("PCA 100:", X_train_pca_100.shape, X_test_pca_100.shape)
print("PCA 200:", X_train_pca_200.shape, X_test_pca_200.shape)
print("LDA (max 9):", X_train_lda.shape, X_test_lda.shape)




