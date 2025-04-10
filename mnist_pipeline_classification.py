import idx2numpy
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import gzip
import time
import matplotlib

#There were issues with main thread not in main from graphics.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1)

# Task 1: Import data
train_images = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
test_images = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
test_labels = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")

#Fashion
train_imagesf = idx2numpy.convert_from_file("train-imagesf-idx3-ubyte")
train_labelsf = idx2numpy.convert_from_file("train-labelsf-idx1-ubyte")
test_imagesf = idx2numpy.convert_from_file("t10k-imagesf-idx3-ubyte")
test_labelsf = idx2numpy.convert_from_file("t10k-labelsf-idx1-ubyte")

# Task 2: flatten images
X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)
y_train = train_labels
y_test = test_labels

#Fashion
X_trainf = train_imagesf.reshape(train_imagesf.shape[0], -1)
X_testf = test_imagesf.reshape(test_imagesf.shape[0], -1)
y_trainf = train_labelsf
y_testf = test_labelsf

#Task 3.1 Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#fashion
scalerf = StandardScaler()
X_train_scaledf = scalerf.fit_transform(X_trainf)
X_test_scaledf = scalerf.transform(X_testf)

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

#Fashion
pca_50f = PCA(n_components=50).fit(X_train_scaledf)
X_train_pca_50f = pca_50f.transform(X_train_scaledf)
X_test_pca_50f = pca_50f.transform(X_test_scaledf)

pca_100f = PCA(n_components=100).fit(X_train_scaledf)
X_train_pca_100f = pca_100f.transform(X_train_scaledf)
X_test_pca_100f = pca_100f.transform(X_test_scaledf)

pca_200f = PCA(n_components=200).fit(X_train_scaledf)
X_train_pca_200f = pca_200f.transform(X_train_scaledf)
X_test_pca_200f = pca_200f.transform(X_test_scaledf)

# LDA examples 
lda_9f = LDA(n_components=9).fit(X_train_scaledf, y_trainf)
X_train_ldaf = lda_9f.transform(X_train_scaledf)
X_test_ldaf = lda_9f.transform(X_test_scaledf)


print("PCA 50:", X_train_pca_50.shape, X_test_pca_50.shape)
print("PCA 100:", X_train_pca_100.shape, X_test_pca_100.shape)
print("PCA 200:", X_train_pca_200.shape, X_test_pca_200.shape)
print("LDA (max 9):", X_train_lda.shape, X_test_lda.shape)

print("PCA 50f:", X_train_pca_50f.shape, X_test_pca_50f.shape)
print("PCA 100f:", X_train_pca_100f.shape, X_test_pca_100f.shape)
print("PCA 200f:", X_train_pca_200f.shape, X_test_pca_200f.shape)
print("LDA (max 9)f:", X_train_ldaf.shape, X_test_ldaf.shape)

confusion_matrices = []
accs = []
accuracy = []

times = []
tm = []

# Task 3.3: SVC with GridSearch
def run_grid_search(X_train, y_train, X_test, y_test, kernel, reduction):
    if kernel == 'linear':
        param_grid = {
            'svc__C': [0.01, 1, 10, 500, 1000, 2000]
        }
        svc = SVC(kernel='linear', max_iter=100000)

    elif kernel == 'rbf':
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': [1e-4, 1e-3, 1e-2, 1e-1]
        }
        svc = SVC(kernel='rbf', max_iter=100000)

    elif kernel == 'poly':
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__gamma': [0.001, 0.01],
            'svc__degree': [2, 3, 4]
        }
        svc = SVC(kernel='poly', max_iter=100000)

    else:
        raise ValueError("Unsupported kernel")

    pipeline = Pipeline([
        ('svc', svc)
    ])

    start = time.time()
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    end = time.time()-start

    print(f"\nBest parameters for {kernel} kernel:", grid.best_params_)
    y_pred = grid.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    accuracy.append(score)
    tm.append(end)
    print(f"Accuracy ({kernel}):", score)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    confusion_matrices.append((confusion_matrix(y_test, y_pred), X_test.shape, reduction, kernel))
    print(f"{kernel} time: {end} seconds")


def plotConfusion(matrices, set):
    for matrix, shape, red, kernel in matrices:
        classes = matrix.shape[0]
        figure, ax = plt.subplots(figsize=(7,5))
        ax.axis('off')
        #Fill in cells with text of prediction values
        cells = [[f'{p}' for p in row] for row in matrix]
        rlab = [f'Y: {i}' for i in range(classes)]
        clab = [f'Y^: {i}' for i in range(classes)]
        table = ax.table(cellText=cells, rowLabels=rlab, colLabels=clab, loc='center')
        plt.title(f"Confusion Matrix, Set: {set}, Features: {shape[1]}, Type: {red}, Kernel: {kernel}")
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.1, 1.3)
        plt.savefig(f'confusion{set}_{shape[1]}_{red}_{kernel}', dpi=300)
        plt.close()

def plotAcc(acc, set):
    figure, ax = plt.subplots(figsize=(7,5))
    ax.axis('off')
    #Fill in cells with text of acc values
    cells = [[f'{a:.2f}' for a in row] for row in acc]
    #Each row is a set
    rlab = ['PCA 50', 'PCA 100', 'PCA 200','FULL', 'LDA 9']
    #columns are kernels
    clab = ['Linear', 'RBF', 'Poly']
    table = ax.table(cellText=cells, rowLabels=rlab, colLabels=clab, loc='center')
    plt.title(f"{set} Accuracy")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(0.9, 1.3)
    plt.savefig(f'set{set}_accuracy.png', dpi=300)
    plt.close()

def plotTime(times, set):
    figure, ax = plt.subplots(figsize=(7,5))
    ax.axis('off')
    #Fill in cells with text of acc values
    cells = [[f'{t:.2f}' for t in row] for row in times]
    #Each row is a set
    rlab = ['PCA 50', 'PCA 100', 'PCA 200', 'FULL', 'LDA 9']
    #columns are kernels
    clab = ['Linear', 'RBF', 'Poly']
    table = ax.table(cellText=cells, rowLabels=rlab, colLabels=clab, loc='center')
    plt.title(f"{set} Runtime (Seconds)")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(0.9, 1.3)
    plt.savefig(f'set{set}_times.png', dpi=300)
    plt.close()        

maxSizeS = 60000
maxSizeT = 10000

sampSize = 3000
testSize = 300
randTrain = np.random.choice(maxSizeS, sampSize, replace=False)
randTest = np.random.choice(maxSizeT, testSize, replace=False)
# Run GridSearch for each kernel
for pca_data, pca_name in zip([(X_train_pca_50, X_test_pca_50), (X_train_pca_100, X_test_pca_100), (X_train_pca_200, X_test_pca_200), (X_train_scaled, X_test_scaled)], ['PCA 50', 'PCA 100', 'PCA 200', 'FULL']):
    print(f"\nRunning GridSearch for {pca_name}...")
    dtrain = pca_data[0]
    dtest = pca_data[1]
    accuracy = []
    tm = []
    for kernel in ['linear', 'rbf', 'poly']:
        run_grid_search(dtrain[randTrain], y_train[randTrain], dtest[randTest], y_test[randTest], kernel, 'PCA')
    accs.append(accuracy)
    times.append(tm)    


# Run GridSearch for LDA transformation
print("\nRunning GridSearch for LDA...")
accuracy = []
tm = []
for kernel in ['linear', 'rbf', 'poly']:
    run_grid_search(X_train_lda[randTrain], y_train[randTrain], X_test_lda[randTest], y_test[randTest], kernel, 'LDA')
accs.append(accuracy)
times.append(tm) 


plotConfusion(confusion_matrices, 'MNIST')
plotAcc(accs, 'MNIST')
plotTime(times, 'MNIST')

confusion_matrices = []
accs = []
times = []

#Grid search for mnist fashion.
for pca_data, pca_name in zip([(X_train_pca_50f, X_test_pca_50f), (X_train_pca_100f, X_test_pca_100f), (X_train_pca_200f, X_test_pca_200f), (X_train_scaledf, X_test_scaledf)], ['PCA 50', 'PCA 100', 'PCA 200', 'FULL']):
    print(f"\nRunning GridSearch for {pca_name}...")
    dtrain = pca_data[0]
    dtest = pca_data[1]
    accuracy = []
    tm = []
    for kernel in ['linear', 'rbf', 'poly']:
        run_grid_search(dtrain[randTrain], y_trainf[randTrain], dtest[randTest], y_testf[randTest], kernel, 'PCA')
    accs.append(accuracy)
    times.append(tm)          

# Run GridSearch for LDA transformation
print("\nRunning GridSearch for LDA...")
accuracy = []
tm = []
for kernel in ['linear', 'rbf', 'poly']:
    run_grid_search(X_train_ldaf[randTrain], y_trainf[randTrain], X_test_ldaf[randTest], y_testf[randTest], kernel, 'LDA')
accs.append(accuracy)
times.append(tm)

plotConfusion(confusion_matrices, "Fashion-MNIST")
plotAcc(accs, "Fashion-MNIST")
plotTime(times, "Fashion-MNIST")
   


