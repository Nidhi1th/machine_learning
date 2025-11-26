#def step(x):
    ##return 1 if x>=0 else 0

#def perceptron(x1, x2, w1, w2, b):
    #y = x1*w1 + x2*w2 + b
    #return step(y)
# (perceptron(0, 0, 1, 1, -1.5))
#print (perceptron(0, 1, 1, 1, -1.5))
#print (perceptron(1, 0, 1, 1, -1.5))
#print (perceptron(1, 1, 1, 1, -1.5))

#  train the preceptron
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=1000, n_features=10, n_classes=2,  random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

#Perceptron model
model = Perceptron(eta0=0.1, max_iter=1000)


clf = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=42,
    tol=1e-3,
    shuffle=True
)

clf.fit(x_test, y_test)

accuracy = clf.score(x_test, y_test)
print(f"The accuracy is{accuracy}")

