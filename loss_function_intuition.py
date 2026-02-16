# Testing how loss functions behave under different fields
import torch
from neural_pde.util.loss_functions import multivariate_normalised_rmse as RMSE
from neural_pde.util.loss_functions import multivariate_normalised_rmse_with_data as RMSE_normalised





ones = torch.ones(4, 3, 17)
perturbation = torch.rand(4, 3, 17)

thousands = 1000 * ones
small = 0.0001 * ones


thousands_perturbed1 = thousands + 2 * perturbation
thousands_perturbed2 = thousands - 2 * perturbation

ones_perturbed1 = ones + perturbation
ones_perturbed2 = ones - perturbation
mean_ones = torch.tensor([1, 1, 1])
std_ones = torch.std(thousands_perturbed1)

mean_thousands = torch.tensor([1000, 1000, 1000])
std_thousands = torch.std(ones_perturbed1)

mixed = ones
mixed[1] = 1000 * mixed[1]

mixed_perturbed1 = mixed + perturbation
mixed_perturbed2 = mixed - perturbation
mean_mixed = torch.tensor([1, 1000, 1])
std_mixed = torch.std(mixed_perturbed1)

# Here, RMSE gives a much smaller error for the thousands perturbed since the error perturbation
# is much smller
print(RMSE(thousands_perturbed1, thousands_perturbed2)) 
print(RMSE(ones_perturbed1, ones_perturbed2))
print(RMSE(mixed_perturbed1, mixed_perturbed2))

# The data is normalised before finding the RMSE. Therefore, since the perturbation is the same,
# these give exactly the same error! 
print(RMSE_normalised(thousands_perturbed1, thousands_perturbed2, mean_thousands, std_thousands)) 
print(RMSE_normalised(ones_perturbed1, ones_perturbed2, mean_ones, std_ones))
print(RMSE_normalised(mixed_perturbed1, mixed_perturbed2, mean_mixed, std_mixed))

# Question: which is the best representation of percentage error?
# RMSE normalised is probably the best, but it might give a higher value?

