import contextlib
import math

import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import distributed as dist

from .networks_edm2 import MPConv

#----------------------------------------------------------------------------
# The forced WN can introduce small numerical errors.
# Disable it in the self teacher step.

@contextlib.contextmanager
def disable_forced_wn(module):
    def set_force_wn(m):
        if isinstance(m, MPConv):
            m.force_wn = False

    def reset_force_wn(m):
            if isinstance(m, MPConv):
                m.force_wn = True

    module.apply(set_force_wn)
    try:
        yield
    finally:
        module.apply(reset_force_wn)

#----------------------------------------------------------------------------
# Loss function in "Consistency Models Made Easy"

@persistence.persistent_class
class ECMLoss:
    def __init__(self, 
            P_mean=-1.1, P_std=2.0, sigma_data=0.5, 
            q=4, c=0.0, k=8.0, b=1.0, adj='sigmoid', wt='snrpk'
            ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
       
        if adj == 'const':
            dist.print0('const adj')
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            dist.print0('sigmoid adj')
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')
        
        if wt == 'snrpk':
            self.wt_fn = self.snrplusk_wt
        else:
            raise ValueError(f'Unknow wt fn type {adj}!')

        self.q = q
        self.stage = 0
        self.ratio = 0.
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'Wt: {wt}, P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)
    
    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def snrplusk_wt(self, t, r):
        # SNR(t) + k = 1/t**2 + k
        wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        return wt

    def __call__(self, net, images, labels=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation
        x_0 = images
        
        # Shared noise direction
        eps = torch.randn_like(x_0)
        x_t = x_0 + eps * t
        x_r = x_0 + eps * r
        
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        fx_t = net(x_t, t, labels)
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                # NOTE(gsunshine): Disable the forced WN since there is no weight update.
                # This eliminates the numerical errors and retains the self consistency.
                with disable_forced_wn(net):
                    fx_r = net(x_r, r, labels)
            
            mask = r > 0
            fx_r = torch.nan_to_num(fx_r)
            fx_r = mask * fx_r + (~mask) * x_0
        else:
            fx_r = x_0

        # L2 Loss
        loss = (fx_t - fx_r) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        # NOTE(gsunshine): Higher p > 0.5 improves first two stages but impede later on ImgNet 64x64. (Further study needed)
        # loss = loss / (loss.detach() + loss_eps) ** p
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        wt = self.wt_fn(t, r)
        return loss * wt.flatten()







@persistence.persistent_class
class ECT2Loss:
    def __init__(self, 
            P_mean=-1.1, P_std=2.0, sigma_data=0.5, 
            q=4, c=0.0, k=8.0, b=1.0, adj='sigmoid', wt='snrpk', gamma=0.5, initial=0.7, mu_type='step_lr'
            ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
       
        if adj == 'const':
            dist.print0('const adj')
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            dist.print0('sigmoid adj')
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        if wt == 'snrpk':
            self.wt_fn = self.snrplusk_wt
        else:
            raise ValueError(f'Unknow wt fn type {adj}!')

        if mu_type == 'step_lr':
            self.mu = self.inv_step_lr
        elif mu_type == 'sigmoid':
            self.mu = self.mu_sigmoid
        else:
            self.mu = self.mu_const

        self.q = q
        self.stage = 0
        self.ratio = 0.

        self.stage_mu = 0
        self.ratio_mu = 0.

        self.gamma = gamma
        self.initial = initial
        
        self.k = k
        self.b = b

        self.c = c
        dist.print0(f'Wt: {wt}, P_mean: {self.P_mean}, P_std: {self.P_std}, q: {self.q}, k {self.k}, b {self.b}, c: {self.c}')

    def update_schedule(self, stage):
        self.stage = stage
        self.ratio = 1 - 1 / self.q ** (stage+1)
    
    def t_to_r_const(self, t):
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (self.stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)
    
    def update_schedule_mu(self, stage_mu):
        self.stage_mu = stage_mu
        self.ratio_mu = 1 - 1 / self.q ** (stage_mu+1)

    def update_schedule_step(self, stage_step):
        self.stage_step = stage_step

    def inv_step_lr(self, t):
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        ratio = 1 - (self.initial * (self.gamma ** self.stage_mu)) * adj
        return torch.clamp(ratio, min=0)

    def mu_const(self, t):
        decay = 1 / 2.0 ** (self.stage_mu+1)
        ratio = 1 - decay
        return torch.clamp(torch.tensor(ratio, dtype=t.dtype), min=0)

    def snrplusk_wt(self, t, r):
        # SNR(t) + k = 1/t**2 + k
        wt = (t ** 2 + self.sigma_data ** 2) / (t * self.sigma_data) ** 2
        return wt
    
    def stf_targets(self, sigmas, perturbed_samples, ref, target):
        """
        Compute stable target using reference batch and noisy samples.

        Args:
            sigmas: noisy levels
            perturbed_samples: perturbed samples with perturbation kernel N(0, sigmas**2)
            ref: the reference batch

        Returns: stable target
        """
        with torch.no_grad():
            perturbed_samples_vec = perturbed_samples.reshape((len(perturbed_samples), -1))
            ref_vec = ref.reshape((len(ref), -1))

            gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - ref_vec) ** 2,
                                    dim=[-1])
            gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
            # adding a constant to the log-weights to prevent numerical issue
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)[:, :, None]
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))

            # target = ref_vec.unsqueeze(0).repeat(len(perturbed_samples), 1, 1)
            #calculate the stable targets with reference batch
            stable_targets = torch.sum(weights * target, dim=1)
            return stable_targets
    
    def get_stf_target_value(self, x_t, ref_images, x_0, t):
            x_t_flat = x_t.view(x_t.size(0), -1)
            x_0_flat = ref_images.view(ref_images.size(0), -1)
            x_0_expanded = x_0_flat.unsqueeze(0).repeat(x_t.size(0), 1, 1)  
            x_t_expanded = x_t_flat.unsqueeze(1).repeat(1, ref_images.size(0), 1)  
            target = (x_t_expanded - x_0_expanded) / t.view(-1, 1, 1)  # (ref_batch, batch_size_t, features)
            stf_target = target.view(x_0.size(0), ref_images.size(0), -1)
            return stf_target
    

    def __call__(self, net, images, labels=None, stf=False, ref_images=None):
        # t ~ p(t) and r ~ p(r|t, iters) (Mapping fn)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(t)

        # Augmentation
        x_0 = images
        
        # Shared noise direction
        eps = torch.randn_like(x_0)
        x_t = x_0 + eps * t
        #x_r = x_0 + eps * r
        # Shared Dropout Mask
        rng_state = torch.cuda.get_rng_state()
        fx_t = net(x_t, t, labels)

        # target score for stf
        if stf:
            stf_target = self.get_stf_target_value(x_t, ref_images, x_0, t)
            target = self.stf_targets(t.squeeze(), x_t, ref_images, stf_target)
            target = target.view_as(x_0)
        else:
            target = x_0
        
        if r.max() > 0:
            torch.cuda.set_rng_state(rng_state)
            with torch.no_grad():
                rt = r / t
                mu = self.mu(t)
                x_r_hat = rt * x_t + (1. - rt) * fx_t
                if stf:
                    x_r_bar = x_t - (t - r) * target
                else:
                    x_r_bar = x_0 + eps * r
                x_r = (1. - mu) * x_r_hat + mu * x_r_bar
                # NOTE(gsunshine): Disable the forced WN since there is no weight update.
                # This eliminates the numerical errors and retains the self consistency.
                with disable_forced_wn(net):
                    fx_r = net(x_r, r, labels)
            
            mask = r > 0
            fx_r = torch.nan_to_num(fx_r)
            fx_r = mask * fx_r + (~mask) * x_0
        else:
            fx_r = x_0

        # L2 Loss
        loss = (fx_t - fx_r) ** 2
        loss = torch.sum(loss.reshape(loss.shape[0], -1), dim=-1)
        
        # Producing Adaptive Weighting (p=0.5) through Huber Loss
        # NOTE(gsunshine): Higher p > 0.5 improves first two stages but impede later on ImgNet 64x64. (Further study needed)
        # loss = loss / (loss.detach() + loss_eps) ** p
        if self.c > 0:
            loss = torch.sqrt(loss + self.c ** 2) - self.c
        else:
            loss = torch.sqrt(loss)
        
        wt = self.wt_fn(t, r)
        return loss * wt.flatten()

