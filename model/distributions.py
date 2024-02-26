import numpy as np
import scipy
import torch
import torch.distributions as D
import ot
from model import FastSpeech2
import itertools

from tqdm import tqdm


class InterpolateGMM(D.MixtureSameFamily):
    def __init__(self, distri_a: D.MixtureSameFamily, distri_b: D.MixtureSameFamily):
        assert type(distri_a.component_distribution) == D.Normal or type(distri_a.component_distribution.base_dist) == D.Normal
        assert type(distri_b.component_distribution) == D.Normal or type(distri_b.component_distribution.base_dist) == D.Normal
        self.distri_a = distri_a.component_distribution if type(distri_a.component_distribution) == D.Normal else distri_a.component_distribution.base_dist
        self.distri_a = (distri_a.mixture_distribution.probs, self.distri_a)
        self.distri_b = distri_b.component_distribution if type(distri_b.component_distribution) == D.Normal else distri_b.component_distribution.base_dist
        self.distri_b = (distri_b.mixture_distribution.probs, self.distri_b)
        self.t = 0.5
        self.ot_Cost = [[self._w2sq(i, j) for j in range(self.distri_b[0].shape[1])] for i in range(self.distri_a[0].shape[1])]
        self.ot_Matrix = ot.emd(self.distri_a[0][0].to("cpu").detach().numpy().copy(), self.distri_b[0][0].to("cpu").detach().numpy().copy(), self.ot_Cost)
        self.mix = D.Categorical(torch.from_numpy(self.ot_Matrix).flatten().unsqueeze(0))
        comp_mu = torch.stack([self._cal_comp_mu(i, j) for j in range(self.distri_b[0].shape[1]) for i in range(self.distri_a[0].shape[1])], 0)
        comp_sigma = torch.stack([self._cal_comp_sigma(i, j) for j in range(self.distri_b[0].shape[1]) for i in range(self.distri_a[0].shape[1])], 0)
        comp = D.Independent(D.Normal(
            comp_mu.unsqueeze(0), comp_sigma.unsqueeze(0)
        ), 1)
        super(InterpolateGMM, self).__init__(self.mix, comp)

    def interpolate_rate(self, t):
        self.t = t
        comp_mu = torch.stack([self._cal_comp_mu(i, j) for j in range(self.distri_b[0].shape[1]) for i in range(self.distri_a[0].shape[1])], 0)
        comp_sigma = torch.stack([self._cal_comp_sigma(i, j) for j in range(self.distri_b[0].shape[1]) for i in range(self.distri_a[0].shape[1])], 0)
        comp = D.Independent(D.Normal(
            comp_mu.unsqueeze(0), comp_sigma.unsqueeze(0)
        ), 1)
        self._component_distribution = comp

    def _cal_comp_mu(self, i, j):
        mu = (self.distri_a[1].mean[0][i], self.distri_b[1].mean[0][j])
        ans = (1 - self.t)*mu[0] + self.t*mu[1]
        return ans

    def _cal_comp_sigma(self, i, j):
        sigma = (torch.diag(self.distri_a[1].variance[0][i]), torch.diag(self.distri_b[1].variance[0][j]))
        sigma = (sigma[0].to("cpu").detach().numpy().copy(), sigma[1].to("cpu").detach().numpy().copy())
        ans = (
            scipy.linalg.inv(scipy.linalg.sqrtm(sigma[0])) * np.linalg.matrix_power
            (
                (1-self.t)*sigma[0] +
                self.t *
                scipy.linalg.sqrtm(
                    scipy.linalg.sqrtm(sigma[0]) *
                    sigma[1] *
                    scipy.linalg.sqrtm(sigma[0])
                ), 2
            ) *
            scipy.linalg.inv(scipy.linalg.sqrtm(sigma[0]))
        )
        ans = torch.diagonal(torch.from_numpy(ans))
        return ans

    def _w2sq(self, i, j):
        mu = (self.distri_a[1].mean[0][i], self.distri_b[1].mean[0][j])
        mu = (mu[0].to("cpu").detach().numpy().copy(), mu[1].to("cpu").detach().numpy().copy())
        sigma = (torch.diag(self.distri_a[1].variance[0][i].flatten()), torch.diag(self.distri_b[1].variance[0][j].flatten()))
        sigma = (sigma[0].to("cpu").detach().numpy().copy(), sigma[1].to("cpu").detach().numpy().copy())
        ans = scipy.linalg.sqrtm(sigma[0])
        ans = -2 * ans * scipy.linalg.sqrtm(
            np.conjugate(ans.T)
            * sigma[1] *
            np.conjugate(ans.T)
        ) * ans
        ans = np.trace(sigma[0] + sigma[1] + ans)
        ans = np.linalg.norm(mu[0]-mu[1]) ** 2 + ans
        return ans

class BarycenterGMM(D.MixtureSameFamily):
    def __init__(self, model: FastSpeech2, device="cpu"):
        self.metadata_list = model.speaker_enc.metadata_list
        self.device = device
        metalist = list(tuple(np.eye(len(metatype))[metaid] for metaid in metatype.values()) for metatype in self.metadata_list.values())
        metalist = self._product(*metalist)
        self.original_distri = {}
        for meta in metalist:
            meta = meta.to(device)
            distri = model.speaker_distribution(meta)
            self.original_distri[meta] = distri.component_distribution if type(distri.component_distribution) == D.Normal else distri.component_distribution.base_dist
            self.original_distri[meta] = {"probs": distri.mixture_distribution.probs, "distributes": self.original_distri[meta]}
        self.rate = [1/len(self.original_distri) for i in range(len(self.original_distri))]
        barycenters = self._barycenter_gaussians(_print=False)
        probs = self._determine_pi(barycenters)
        comp_mean, comp_std = torch.stack([barycenters[use]["mean"] for use in probs.keys()], 0).unsqueeze(0), torch.stack([barycenters[use]["std"] for use in probs.keys()], 0).unsqueeze(0)
        comp = D.Independent(D.Normal(
            comp_mean, comp_std
        ), 1)
        mix = D.Categorical(torch.tensor([v for v in probs.values()]).flatten().unsqueeze(0).to(device))
        super(BarycenterGMM, self).__init__(mix, comp)

    
    def _product(self, *args, repeat=1):
        pools = [pool for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [np.concatenate([x, y], 0) for x in result for y in pool]
        for prod in result:
            yield torch.tensor(prod, dtype=torch.float).view(1, -1)
    
    def barycenter_rate(self, rate, _print=True):
        assert hasattr(rate, "__len__") and len(rate) == len(self.original_distri)
        assert sum(rate) == 1
        self.rate = rate
        if(_print):
            print("rate: ", rate)
            for i, distri in enumerate(self.original_distri):
                print("distribution " + str(i+1) + "(rate: " + str(rate[i]) + ")")
                point = 0
                for meta in self.metadata_list.items():
                    print(" " + meta[0] + ":")
                    for k in meta[1]:
                        print("     " + k + ": " + str(distri[0][point].item()))
                        point += 1
        barycenters = self._barycenter_gaussians()
        probs = self._determine_pi(barycenters)
        comp_mean, comp_std = torch.stack([barycenters[use]["mean"] for use in probs.keys()], 0).unsqueeze(0), torch.stack([barycenters[use]["std"] for use in probs.keys()], 0).unsqueeze(0)
        comp = D.Independent(D.Normal(
            comp_mean, comp_std
        ), 1)
        mix = D.Categorical(torch.tensor([v for v in probs.values()]).flatten().unsqueeze(0).to(self.device))
        self._component_distribution = comp
        self._mixture_distribution = mix
        
    
    
    def _barycenter_gaussians(self):
        tqdm.write("..calculating barycenters..")
        barycenters = {}
        pos_list = list(itertools.product(*[list(range(d["probs"].shape[1])) for d in self.original_distri.values()]))
        for pos in tqdm(pos_list):
            barycenters[pos] = self._one_barycenter_gaussian(self.rate, pos)
        return barycenters

    def _one_barycenter_gaussian(self, rate, pos: tuple):
        assert len(pos) == len(self.original_distri) and len(rate) == len(pos)
        gaussians = [None] * len(pos)
        for i, d in enumerate(self.original_distri.values()):
            gaussians[i] = d["distributes"]
        
        mean = sum([rate[i] * gaussians[i].mean[0][pos[i]] for i in range(len(pos))])

        original_std = tuple(gaussians[i].stddev[0][pos[i]] for i in range(len(pos)))
        std = original_std[0].clone()
        for i in range(60):
            std = (
                (1 / std)*
                    sum(
                        [
                            (rate[j] * std * original_std[j]) for j in range(len(pos))
                        ]
                    )
            )
        return {"mean": mean, "std": std}
    
    def _determine_pi(self, barycenters: dict):
        tqdm.write("..calculating probs..")
        probs = {}
        for i, dist in tqdm(enumerate(self.original_distri.values())):
            rate = self.rate[i]
            for j, prob in tqdm(enumerate(dist["probs"].view(-1)), leave=False):
                n_distri = {"mean": dist["distributes"].mean[0][j], "std": dist["distributes"].stddev[0][j]}
                min_distance = None
                min_pos = None
                for pos, b_distri in barycenters.items():
                    if min_distance == None:
                        min_distance = self._w2sq(n_distri, b_distri)
                        min_pos = pos
                        continue
                    distance = self._w2sq(n_distri, b_distri)
                    if distance < min_distance:
                        min_distance = distance
                        min_pos = pos
                probs[min_pos] = rate * prob.item() if min_pos not in probs else probs[min_pos] + rate*prob.item()
        return probs

    def _w2sq(self, distri_a, distri_b):
        mu = (distri_a["mean"].flatten(), distri_b["mean"].flatten())
        sigma = (distri_a["std"].flatten(), distri_b["std"].flatten())
        ans = torch.norm(mu[0]-mu[1])**2 + sum(
            (sigma[0][i] - sigma[1][i]) ** 2 for i in range(len(sigma[0]))
        )
        return ans