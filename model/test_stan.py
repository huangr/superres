import pystan

model_code = open("res.stan").read()
fit = pystan.stan(model_code=model_code)

print fit

