from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_digits,y_digits = load_digits(return_X_y=True)
print(X_digits.shape)
print(X_digits)
print(y_digits)
X_train,X_test,y_train,y_test = train_test_split(X_digits,y_digits)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print(log_reg.score(X_test, y_test))



