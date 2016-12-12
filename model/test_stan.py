import pystan
from stan_inputs import *

file_path = "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/grayscale/alpaca"

img_data = define_variables(file_path, 15, 35, 400)

fit = pystan.stan(file="res.stan", data=img_data)

print fit

