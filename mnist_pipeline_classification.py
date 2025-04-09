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

# Task 3.3: SVC with GridSearch
def run_grid_search(X_train, y_train, X_test, y_test, kernel):
    if kernel == 'linear':
        param_grid = {
            'svc__C': [0.01, 0.1, 1, 10, 100, 500, 1000, 2000]
        }
        svc = SVC(kernel='linear')

    elif kernel == 'rbf':
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': [1e-4, 1e-3, 1e-2, 1e-1]
        }
        svc = SVC(kernel='rbf')

    elif kernel == 'poly':
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': [0.001, 0.01],
            'svc__degree': [2, 3, 4]
        }
        svc = SVC(kernel='poly')

    else:
        raise ValueError("Unsupported kernel")

    pipeline = Pipeline([
        ('svc', svc)
    ])

    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\nBest parameters for {kernel} kernel:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print(f"Accuracy ({kernel}):", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))



