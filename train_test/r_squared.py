import numpy
from sklearn.metrics import r2_score


numpy.random.seed(2)
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
my_model = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
r2_training_set = r2_score(train_y, my_model(train_x))
r2_testing_set = r2_score(test_y, my_model(test_x))
print(r2_training_set)
print(r2_testing_set)
