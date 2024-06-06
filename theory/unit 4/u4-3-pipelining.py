from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

X_digits,y_digits = load_digits(return_X_y=True)
print(X_digits.shape)
print(X_digits)
print(y_digits)
X_train,X_test,y_train,y_test = train_test_split(X_digits,y_digits)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print(log_reg.score(X_test, y_test))


pipeline = Pipeline([
    ("kmeans",KMeans(n_clusters=50,n_init=100)),
    ("log_reg",LogisticRegression()),
    # ("log_reg",LogisticRegression(solver='lbfgs', max_iter=1000)),
])

pipeline.fit(X_train, y_train)
print(pipeline.score(X_test,y_test))


# cross validation for better k value or 
# use the following 
# verboseint
# Controls the verbosity: the higher, the more messages.
# >1 : the computation time for each fold and parameter candidate is displayed;
# >2 : the score is also displayed;
# >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.

from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2,100))
grid_clf = GridSearchCV(pipeline,param_grid,cv=3,verbose=2)
grid_clf.fit(X_train, y_train)

print(grid_clf.best_params_)
print(grid_clf.score(X_test, y_test))




