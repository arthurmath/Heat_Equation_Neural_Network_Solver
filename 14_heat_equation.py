import numpy as np
import time
import os
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, InputLayer, Dropout, Input #, BatchNormalization, preprocessing 
#from tensorflow.python.keras.optimizers import Adam, SDG
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, SGDRegressor, ElasticNet, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor



# solve AttributeError: module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'.
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset


## Reproducability
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
    return seed
seed = set_seed(42)


## FINITE ELEMENTS MATRIX
def get_K(Nx, Ny, L, alpha):
    """ Returns stiffness matrix K """
    dx = L / Nx
    dy = L / Ny
    N = Nx * Ny
    K = sp.sparse.lil_matrix((N, N))
    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j
            
            if i < Nx-1:
                right = idx + Ny
                K[idx, idx] += alpha / dx
                K[idx, right] -= alpha / dx
                K[right, idx] -= alpha / dx
                K[right, right] += alpha / dx

            if j < Ny-1:
                top = idx + 1
                K[idx, idx] += alpha / dy
                K[idx, top] -= alpha / dy
                K[top, idx] -= alpha / dy
                K[top, top] += alpha / dy
    K += 0.1 * np.eye(N) # regularization
    return K


## HEAT FLUX
def flux(Nx, Ny, x, y, value):
    f = np.zeros((Nx, Ny))
    f[x, y] = value
    f = f.flatten()
    return f


## PARAMETERS
Nx = 20  # Number of points in x-direction
Ny = 20  # Number of points in y-direction
N = Nx * Ny # Number of DOFs
L = 1.0  # Length of the square domain
alpha = 1.0  # Thermal diffusivity
K = get_K(Nx, Ny, L, alpha)


# ## EXAMPLE 
# f = flux(Nx, Ny, x=3, y=6, value=-3)
# u = sp.sparse.linalg.spsolve(K.tocsr(), f).real

# fig, ax = plt.subplots(figsize=(9, 5))
# field = u.reshape((Nx, Ny)).T
# cax_heat = plt.imshow(field, cmap='plasma', interpolation='nearest')
# plt.title('Champ de temperature')
# ax.set_xlabel("Axe x")
# ax.set_ylabel("Axe y")
# plt.colorbar(cax_heat)
# plt.show()



## DATASET CREATION
def dataset(num):
    """ Unique flux """
    X = []
    Y = []
    for value in tqdm(list(np.linspace(-3, 3, num=num, endpoint=True)), desc="Processing items", unit="item"): 
        for x in range(Nx):
            for y in range(Ny):
                f = flux(Nx, Ny, x, y, value)
                u = sp.sparse.linalg.spsolve(K.tocsr(), f).real 
                X.append(f)
                Y.append(u)
    return X, Y

def dataset2(num):
    """ Multi flux """
    X = [] 
    Y = [] 
    for _ in tqdm(range(N*num), desc="Processing items", unit="item"): 
        f = np.zeros(N)
        num_heat_sources = np.random.randint(1, 10)  # Number of heat sources
        positions = np.random.choice(N, num_heat_sources, replace=False)
        values = np.random.rand(num_heat_sources) * 10  # Heat flux values
        f[positions] = values
        u = sp.sparse.linalg.spsolve(K.tocsr(), f).real 
        X.append(f)
        Y.append(u)
    return X, Y



K = get_K(Nx, Ny, L, alpha)
print("\nStarting dataset generetion")
start = time.perf_counter()
X, Y = dataset2(100)
end = time.perf_counter()
exe_time = end - start          
print(f"Execution time: {exe_time:.2f}s") 
print(np.shape(X), np.shape(Y))

X = StandardScaler().fit_transform(X)
Y = StandardScaler().fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, shuffle=True)







## MODELS

## TENSORFLOW
# model = keras.Sequential([
#     # # Preprocessing for Data augmentation
#     # preprocessing.RandomContrast(factor=0.5), # contrast change by up to 50%
#     # preprocessing.RandomFlip(mode='horizontal'), # flip left-to-right
#     # preprocessing.RandomFlip(mode='vertical'), # flip top-to-bottom
#     # preprocessing.RandomWidth(factor=0.15), # horizontal stretch
#     # preprocessing.RandomRotation(factor=0.10),
#     # preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
#     # Bloc 1
#     #InputLayer(shape=[N]),
#     Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(Nx, Ny)),
#     Conv2D(32, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     # Bloc 2
#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     Conv2D(64, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),
#     # Bloc 3
#     Conv2D(128, (3, 3), padding='same', activation='relu'),
#     Conv2D(128, (3, 3), padding='same', activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.25),

#     # Couches de classification
#     Flatten(),
#     Dense(150, activation='relu'),
#     Dropout(0.5),
#     Dense(100, activation='softmax'), #'sigmoid'
# ])

# model = Sequential([
#     Input(shape=(N,)),
#     Dense(256, activation='linear'),
#     #Dropout(0.3),
#     Dense(256, activation='linear'),
#     #Dropout(0.3),
#     #Dense(256, activation='linear'),
#     #Dense(256, activation='linear'), # activation='tanh', 'linear', 'sigmoid'
#     Dense(N)
# ]) 

# mysgd = SGD(learning_rate=0.001, decay=1e-7, momentum=0.9, nesterov=True)
# mysgd = Adam(learning_rate=0.001, decay=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) #metrics=['mae']

# early_stopping = keras.callbacks.EarlyStopping(
#     patience=5, min_delta=0.001, monitor='loss',
#     restore_best_weights=True,
# )

# rp = keras.callbacks.ReduceLROnPlateau(
#     monitor="loss", factor=0.2,
#     patience=3, verbose=1,
#     mode="max", min_lr=0.00001,
# )

# print(model.summary())


## SCIKIT-LEARN
model = LinearRegression(n_jobs=-1) # best precision




## RESULTS :
# MSE relu 0.00205
# MSE sigmoid 0.02328
# MSE linear 5 Dense 0.000152
# MSE linear 3 Dense : 8.29e-05 

# mutliflux :
# MSE linear : 0.0088
# MSE tanh : 0.133
# MSE sigmoid : 0.0928

# ExtraTreesRegressor : 0.708
# RandomForestRegressor : 0.675
# LinearRegression : 2.57e-15    
# Ridge : 1.94e-4
# RidgeCV : 1.94e-5
# ElasticNet : 0.997
# MultiTaskElasticNet : 0.786
# Lasso : 0.997
# GaussianProcessRegressor : 0.786
# KNeighborsRegressor : 0.907
# Xgboost : too long (0.64)








## TRAINING
print("\nModel training")
start = time.perf_counter()
history = model.fit(
    X_train, Y_train,)
    #validation_data=(X_test, Y_test),
    #batch_size=N, #512, 32
    #epochs=100, #200
    #callbacks=[early_stopping, rp])
end = time.perf_counter()
print("End train")
exe_time = end - start          
print(f"Execution time: {exe_time:.2f}s\n") 


# model.save_weights('Machine_Learning/heat.weights.h5')
# model.load_weights('Machine_Learning/heat.weights.h5')

# plt.plot(history.history['loss'])




### EVALUATION
score = model.score(X_test, Y_test)
print("Model's R2 score :", score)

Y_predict = model.predict(X_test) 
print('FINAL RMSE:', root_mean_squared_error(Y_test, Y_predict), '\n')



## PLOT RANDOM PREDICTIONS
fig, ax = plt.subplots(2, 4, figsize=(16, 7))

for idx, i in enumerate([31, 47, 122, 400]):
    plt.subplot(241 + 2*idx)
    cax_heat = plt.imshow(Y_test[i].reshape((Nx, Ny)).T, cmap='plasma', interpolation='nearest')
    plt.title('Real heat field')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.colorbar(cax_heat)

    plt.subplot(242 + 2*idx)
    cax_heat = plt.imshow(Y_predict[i].reshape((Nx, Ny)).T, cmap='plasma', interpolation='nearest')
    plt.title('Predicted heat field')
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.colorbar(cax_heat)

plt.show()





## EXECUTION TIMES
import timeit
f = X_test[120].reshape(-1, 1)

exe_time_inv = np.array(timeit.repeat('np.linalg.inv(K.toarray()) @ f', globals=globals(), number=100, repeat=100)) / 1e2
exe_time_sol = np.array(timeit.repeat('np.linalg.solve(K.toarray(), f)', globals=globals(), number=100, repeat=100)) / 1e2
exe_time_spa = np.array(timeit.repeat('sp.sparse.linalg.spsolve(K.tocsr(), f)', globals=globals(), number=100, repeat=100)) / 1e2
exe_time_mod = np.array(timeit.repeat('model.predict(f.reshape(1, -1))', globals=globals(), number=100, repeat=100)) / 1e2


strings = ['inversion: ', 'solve :    ', 'sparse :   ', 'model :    ']
times = [exe_time_inv, exe_time_sol, exe_time_spa, exe_time_mod]

for str, tim in zip(strings, times):
    print(f"Execution time {str}: {(np.mean(tim)/1e-3):.3f} ms, variance : {(np.var(tim)/1e-9):.2f} ns")
print("")


### Voir PINN