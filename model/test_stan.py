import pystan
from stan_inputs import *

file_path = "/Users/rickh/Dropbox/MIT/6.867/867_project/datasets/grayscale/face1"

img_data, img_inputs = define_variables(file_path, 3, 6, 400)
chain_num = 4
fit = pystan.stan(file="res.stan", data=img_data, iter=100, init=[img_inputs for i in range(chain_num)], chains=chain_num)

print fit

