# Heat_Equation_Neural_Network_Solver

Python Program solving Heat equation with a Neural Network. It begins with the creation of an artificial database. On a 2D square domain, we create up to five heat flux randomly placed on it. We create the associated heat field by computing the finite elements matrix and then solving the system with a classical library (Scipy sparse). We have at the end two lists of matrices, X which contains heats flux on the domain, and Y which contains heat field solutions. 

We implemented many models, from Tensorflow Sequential with Preproccessing (contrast, flips, rotations...), Convolution, Max pooling and Dense layers. We also tested many Scikit-Learn models, like LinearRegression, Ridge, RidgeCV, Lasso, Gradient Boosting, SVM... We separate the dataset in a train and a test datasets and we then train the model. We calculate their R2 scores and RMSE relative to the test dataset. Finally we plot some randomly chosen fields to compare them with the solutions. 

One interesting point is that when we calcuate the inference time of the model, it is 1/3 faster than the classical matrix inversion. We calculate those execution times at the end of the program with timeit.
