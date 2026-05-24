# Heat Equation Neural Network Solver

Python program solving Heat equation with a Neural Network. It begins with the creation of an artificial database. On a 2D square domain, we create up to ten heat sources randomly placed and with random values. We create the associated heat field by computing the finite elements matrix and then solving the system with a classical library (Scipy sparse). We have at the end two lists of matrices, X which contains heats flux on the domain, and Y which contains heat field solutions. 

We implemented many models, from Tensorflow Sequential with Preproccessing (contrast, flips, rotations...), Convolution, Max pooling and Dense layers. We also tested many Scikit-Learn models, like LinearRegression, Ridge, RidgeCV, Lasso, Gradient Boosting, SVM... We separate the dataset in a train and a test datasets and we then train the model. We calculate their R2 scores and RMSE relative to the test dataset. Finally we plot some randomly chosen fields to compare them with the solutions. Surprisingly, the model with higher precision is Sk-Learn's Linear Regression.

One interesting point is that the inference time of the model is 1/3 faster than the classical matrix resolution. We calculated those execution times at the end of the program with timeit module.

<img width="1272" alt="Capture d’écran 2024-11-13 à 18 27 14" src="https://github.com/user-attachments/assets/d5efb419-e3be-458a-bae5-ad856425815e">

Figure: Four randomly chosen heat fields calculated by finite elements matrix resolution or with a Machine Learning model. The best model gave a relative error in the order of 10^(-15).
