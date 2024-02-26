import torch
import numpy as np
import scipy
from scipy import linalg

pos=[1, 2, 3]
original_mean = ([5 ,4, 2], [4, 5, 6])
original_mean = tuple(torch.tensor(s, dtype=torch.float64) for s in original_mean)
original_std = ([5.3, 8, 4], [0, 1, 0])
original_std = tuple(torch.tensor(s, dtype=torch.float64) for s in original_std)
std = original_std[0].clone()

mu = original_mean
sigma = original_std
w2 = torch.norm(mu[0]-mu[1])**2 + sum(
    (sigma[0][i] - sigma[1][i]) ** 2 for i in range(len(sigma[0]))
)
print(w2)

mu = (mu[0].to("cpu").detach().numpy().copy(), mu[1].to("cpu").detach().numpy().copy())
sigma = (torch.diag(original_std[0]*original_std[0]), torch.diag(original_std[1]*original_std[1]))
sigma = (sigma[0].to("cpu").detach().numpy().copy(), sigma[1].to("cpu").detach().numpy().copy())
ans = scipy.linalg.sqrtm(sigma[0])
ans = -2 * ans * scipy.linalg.sqrtm(
    np.linalg.pinv(ans)
    * sigma[1] *
    np.linalg.pinv(ans)
) * ans
ans = np.trace(sigma[0] + sigma[1] + ans)
ans = np.linalg.norm(mu[0]-mu[1]) ** 2 + ans
print(ans)