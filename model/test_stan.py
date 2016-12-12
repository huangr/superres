import pystan
from stan_inputs import *

file_path = "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/grayscale/alpaca"

img_data, img_inputs = define_variables(file_path, 9, 15, 400)

fit = pystan.stan(file="res.stan", data=img_data, iter=10, init=[img_inputs for i in range(4)])

print fit

