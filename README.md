# Elsevier_ML
Codes and source files for the application of predictive methods to estimate density and speed of sound in ionic liquids

The repository contains three scripts that deal with:
- Spinodal Method;
- Neural Network;
-The XGBoost Method

It should be noted that the Spinodal method has a radically different algorithmic basis than NN/ XGBoost. In the case of the Neural Network and XGBoost method, it is enough to replace the example learning database 'zbior23.csv' with, for example, 'densities-database.csv' and make very easy changes in the code to also obtain densities analogously. 
Mathematically, we turn the function f(X) -> u(p,T) into the function f(X) -> rho(p,T)

For the XGBoost method, just replace the method with, for example, from.sklearn import sklearn.ensemble.GradientBoostingRegressor and do the calculation. The same situation applies to methods such as K-means, SVR, etc. 

---------
The structure of the code is designed so that the script can be run immediately in an IDE (e.g. PyCharm) or after copying in a JuPyter notepad file. With such an uncomplicated structure, we recommend using the above scripts in JuPyter, the reason being that it is easy to analyze the code step-by-step, without having to run the whole thing. 

If more methods are added, we plan to successively modify the structure of the code while maintaining its functionality: 
- model files defined as a separate class;
-A file that splits, parses, normalizes data into training and learning;
-The main.py file, which, depending on the need: downloads the database, prepares the data (from the data_preprocess.py file), then runs the corresponding model and returns the data.

However, for the moment, the code has full functionality, and we believe that introducing an OOP approach is unnecessary.
