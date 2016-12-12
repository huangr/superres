import pystan

fit = pystan.stan(file="res.stan")

print fit

