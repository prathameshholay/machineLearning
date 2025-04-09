#starting with importing required packages which is 1.scikit-learn's LinearRegression functions
# which basically learns patter like y = 2x + 3

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data: x -> y (we secretly use y = 2x + 3)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

# Create a model
model = LinearRegression()

# "Train" the machine
model.fit(X, y)

# Now ask it to predict something
prediction = model.predict([[6]])

print("The machine predicts y for x=6 as:", prediction[0])
print("The machine 'learned' the formula: y =", model.coef_[0], "* x +", model.intercept_)


#EXPLAINATION
"""
🧠 How Machines Learn - Simple Example

Imagine you give someone these pairs of numbers:
    x: 1 → y: 5
    x: 2 → y: 7
    x: 3 → y: 9

Now you ask them:
    "What do you think y would be if x = 6?"

This is basically what machines do. They look at patterns in the data and try to learn the relationship.
In this example, we’ll teach a machine to learn that y = 2x + 3.
"""
