
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay 


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

predicted=neigh.predict(X_test)
print(confusion_matrix(y_test, predicted))
cm=confusion_matrix(y_test, predicted)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show() 
