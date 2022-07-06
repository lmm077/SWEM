import math
import torch
from torch import nn
from torch.nn import functional as F


def l2norm(inp, dim):
    norm = torch.linalg.norm(inp, dim=dim, keepdim=True) + 1e-6
    return inp/norm


# use GLU(Gated Linear Unit) for feature fusion
class FeatureFusionLayer(nn.Module):
    def __init__(self, indim, outdim):
        super(FeatureFusionLayer, self).__init__()
        self.layer_f = nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1)
        self.layer_a = nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1)

        nn.init.orthogonal_(self.layer_f.weight.data)
        nn.init.zeros_(self.layer_f.bias.data)

        nn.init.orthogonal_(self.layer_a.weight.data)
        nn.init.zeros_(self.layer_a.bias.data)

    def forward(self, x):
        return self.layer_f(x) * torch.sigmoid(self.layer_a(x))


class MemoryBank:
    def __init__(self, mode='updated'):
        # parameters
        # mode {'fixed', 'slow', 'fast'}
        self.mode = mode
        assert (self.mode in ['fixed', 'updated'])
        self.bases = None
        # number of objects
        self.n_objs = 0

    def initial_memory(self):
        self.bases = None
        # number of objects
        self.n_objs = 0

    def add_new(self, bases):
        if self.bases is None:
            self.bases = bases
            self.n_objs = bases['kappa'].shape[1]
        else:
            N = bases['kappa'].shape[1]
            if N > self.n_objs:
                for key in bases.keys():
                    self.bases[key] = torch.cat([self.bases[key], bases[key][:, self.n_objs:]], dim=1)
            self.n_objs = N

    def update(self, bases):
        # add new objects if have
        if self.mode == 'fixed':
            self.add_new(bases)
        else:
            self.bases = bases


class SWEMCore(nn.Module):
    def __init__(self, n_bases=256, valdim=512, n_iters=4, tau=0.05, topl=64):
        super(SWEMCore, self).__init__()

        self.n_bases = n_bases
        self.n_iters = n_iters
        self.tau = tau
        assert (self.tau > 0)
        self.valdim = valdim

        # memory
        self.memories = dict()
        self.memories['first'] = MemoryBank(mode='fixed')
        self.memories['update'] = MemoryBank(mode='updated')

        # memory random drop
        self.p_drop = 0.0
        # topk for selected mems
        self.topl = int(min(self.n_bases, topl))

        # feature fusion
        self.fusion_layer = FeatureFusionLayer(valdim * 2 + self.topl * 2, valdim)

    def empty(self):
        for key in self.memories.keys():
            self.memories[key].initial_memory()

    ############################################################################
    ### construct low-rank features for fg and bg
    ############################################################################
    @torch.no_grad()
    def sww_step(self, kappa, x_t, masks):
        # kappa, (B, N, 2, Ck, L)
        # x_t,  (B, 1, 1, HW, Ck)
        # masks, (B, N, 2, HW, 1), bg&fg
        x_t_normed = l2norm(x_t, dim=-1)
        kappa_normed = l2norm(kappa, dim=-2)
        z = torch.matmul(x_t_normed, kappa_normed)        # B, N, 2, HW, L
        maxes = torch.max(z, dim=-1, keepdim=True)[0]     # B, N, 2, HW, 1
        maxes = torch.max(maxes, dim=2, keepdim=True)[0]  # B, N, 1, HW, 1
        z_exp = torch.exp((z - maxes) / self.tau)       # B, N, 2, HW, L
        # mean
        sum_exp = torch.sum(z_exp, dim=-1, keepdim=True)         # B, N, 2, HW, 1
        # getting foreground and background probabilities of all pixels
        props = sum_exp / (torch.sum(sum_exp, dim=2, keepdim=True))  # B, N, 2, HW, 1
        weights = masks * (1-props)
        
        return weights

    @torch.no_grad()
    def swe_step(self, x_t, kappa, weights):
        # MEM with single step
        kappa_normed = l2norm(kappa, dim=-2)
        z = torch.matmul(x_t, kappa_normed)                 # B, N, 2, HW, L
        max_z = torch.max(z, dim=-1, keepdim=True)[0]
        z = F.softmax((z - max_z) / self.tau, dim=-1)
        z = z * weights                                     # weighted
        return z

    @torch.no_grad()
    def swm_step(self, z, x, kappa_, zita_):
        # MEM with single step
        zita = zita_ + z.sum(dim=-2, keepdim=True)
        kappa = (zita_ * kappa_ + torch.matmul(x, z)) / zita
        return kappa, zita

    def swem(self, x, v, masks, bases_=None):
        # x, (B, Ck, H, W)
        # v, (B, N, Cv, H, W)
        # kappa, (B, N, 2, Ck, L)
        # masks, (B, N, 2, H, W)
        B, Ck, H, W = x.shape
        N = masks.shape[1]
        if bases_ is None:
            kappa_, nu_, zita_ = self.random_init(size=(B, N, 2, Ck, self.n_bases), dtype=x.type(), device=x.device)
        else:
            kappa_, nu_, zita_ = bases_['kappa'], bases_['nu'], bases_['zita']
        N_new = N - kappa_.shape[1]
        if N_new > 0:
            new_kappa, new_nu, new_zita = self.random_init(size=(B, N_new, 2, Ck, self.n_bases),
                                                           dtype=x.type(), device=x.device)
            kappa_ = torch.cat([kappa_, new_kappa], dim=1)
            nu_ = torch.cat([nu_, new_nu], dim=1)
            zita_ = torch.cat([zita_, new_zita], dim=1)

        b, n, _, h, w = masks.shape
        x = x.flatten(start_dim=-2).unsqueeze(1).unsqueeze(2)  # B, 1, 1, Ck, HW
        masks = masks.flatten(start_dim=-2).unsqueeze(-1)  # B, N, 2, HW, 1
        x_t = x.transpose(-2, -1)  # B, 1, 1, HW, Ck
        weights = masks.clone()
        kappa = kappa_.clone()

        for i in range(self.n_iters):
            # E step
            z = self.swe_step(x_t, kappa, weights)
            # M step
            kappa, zita = self.swm_step(z, x, kappa_, zita_)
            # W step
            if i < self.n_iters - 1:
                weights = self.sww_step(kappa, x_t, masks)

        mv = v.flatten(start_dim=-2).unsqueeze(2)     # B, N, 1, Cv, HW
        nu = (zita_ * nu_ + torch.matmul(mv, z)) / zita

        bases = {'kappa': kappa, 'nu': nu, 'zita': zita}
        return bases

    def random_init(self, size, norm_dim=-2, dtype=None, device=None):
        B, N, _, _, L = size
        kappa = torch.zeros(size=size).type(dtype).to(device)
        kappa.normal_(0, math.sqrt(2. / size[-1]))
        kappa = l2norm(kappa, dim=norm_dim)
        nu = torch.zeros(B, N, 2, self.valdim, L).type(dtype).to(device)
        zita = torch.zeros(B, N, 2, 1, L).type(dtype).to(device) + 1e-6

        return kappa, nu, zita

    ############################################################################
    ### memorize features
    ############################################################################
    def memorize(self, qk, qv, masks):
        if self.memories['update'].bases is None:
            bases = self.swem(qk, qv, masks, self.memories['first'].bases)
        else:
            bases = self.swem(qk, qv, masks, self.memories['update'].bases)

        if self.memories['first'].bases is None:
            self.memories['first'].update(bases)
        else:
            self.memories['first'].update(bases)
            self.memories['update'].update(bases)

    ############################################################################
    ### matching
    ############################################################################
    def perm_inv_feat(self, affinity):
        # permutation invariance, affinity after exp, BN, 2, L, H, W
        sorted_aff = torch.topk(affinity, k=self.topl, dim=2)[0]  # BN, 2, k, H, W
        feat = torch.zeros_like(sorted_aff)
        feat[:, :, 0] = sorted_aff[:, :, 0]
        for i in range(1, self.topl):
            feat[:, :, i] = feat[:, :, i-1] + sorted_aff[:, :, i]

        feat = feat[:, 0] / (feat[:, 0] + feat[:, 1])              # BN, k, H, W
        feat = torch.cat([feat, 1-feat], dim=1)                    # BN, 2k, H, W; [bg, fg]
        return feat

    @torch.no_grad()
    def gen_kernels(self, qk, mk, affinity, n_kernel=7, sigma=7):
        B, Ck, H, W = qk.shape
        N, L = mk.shape[1], mk.shape[-1]
        topk_scores, topk_idxes = torch.topk(affinity, k=n_kernel, dim=-1)  # B, N, 2, L, k
        x_idxes = topk_idxes % W
        y_idxes = (topk_idxes // W) % H
        x_idxes = x_idxes.unsqueeze(-2)  # B, N, 2, L, 1, k
        y_idxes = y_idxes.unsqueeze(-2)  # B, N, 2, L, 1, k

        yv, xv = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # H, W
        yv = yv.float().to(qk.device)
        yv = yv.view(1, 1, 1, 1, H * W, 1).expand(B, N, 2, L, -1, -1).contiguous()  # B, N, 2, L, HW, 1
        xv = xv.float().to(qk.device)
        xv = xv.view(1, 1, 1, 1, H * W, 1).expand(B, N, 2, L, -1, -1).contiguous()  # B, N, 2, L, HW, 1

        gausses = - ((xv - x_idxes) ** 2 + (yv - y_idxes) ** 2) / (2 * sigma ** 2)  # B, N, 2, L, HW, k
        gauss_max = torch.max(gausses, dim=-1, keepdim=False)[0]  # B, N, 2, L, HW
        exp_gauss = torch.exp(gauss_max / self.tau)

        return exp_gauss

    def get_affinity(self, qk, mk, mv, n_kernel=0, sigma=7):
        """
        :param qk: Query key, B, Ck, H, W
        :param mk: Memory key, B, N, 2, Ck, L
        :param mv: Memory value, B, N, 2, Cv, L
        :param n_kernel: number of Gaussion kernels
        :param sigma: std in Gaussion kernel
        :return: kernelized affinity
        """

        B, Ck, H, W = qk.shape
        N, L = mk.shape[1], mk.shape[-1]
        Cv = mv.shape[-2]

        qk_ = qk.flatten(start_dim=-2).unsqueeze(1).unsqueeze(2)   # B, 1, 1, C, HW
        affinity = torch.matmul(mk.transpose(-2, -1), qk_)         # B, N, 2, L, HW, [-1, 1]
        maxes = torch.max(affinity, dim=2, keepdim=True)[0]        # B, N, 1, L, HW, [-1, 1]
        maxes = torch.max(maxes, dim=3, keepdim=True)[0]           # B, N, 1, 1, HW, [-1, 1]
        exp_aff = torch.exp((affinity - maxes) / self.tau)       # B, N, 2, L, HW

        # mkm only valid during inference
        if n_kernel > 0 and not self.training:
            exp_gauss = self.gen_kernels(qk, mk, affinity, n_kernel, sigma)
            exp_aff_gauss = (exp_aff * exp_gauss).flatten(start_dim=2, end_dim=3)
            p_aff = exp_aff_gauss / (torch.sum(exp_aff_gauss, dim=2, keepdim=True) + 1e-8)         # B, N, 2L, HW
        else:
            if self.p_drop > 0 and self.training:
                random_mask = torch.rand(B, N, 1, L, 1).type(exp_aff.dtype).to(exp_aff.device)
                drop_mask = (random_mask > self.p_drop).type_as(random_mask)
                exp_aff_droped = exp_aff * drop_mask
                p_aff = exp_aff_droped / (torch.sum(exp_aff_droped, dim=[2, 3], keepdim=True) + 1e-6)  # B, N, 2, L, HW
                p_aff = p_aff.flatten(start_dim=2, end_dim=3)                                          # B, N, 2L, HW
            else:
                p_aff = exp_aff / torch.sum(exp_aff, dim=[2, 3], keepdim=True)  # B, N, 2, L, HW
                p_aff = p_aff.flatten(start_dim=2, end_dim=3)                   # B, N, 2L, HW

        # affinity features
        exp_aff = exp_aff.view(B * N, 2, -1, H, W)                               # BN, 2, L, H, W
        S = self.perm_inv_feat(exp_aff)                                          # BN, L, H, W
        # aggregation values
        mv = mv.transpose(2, 3).flatten(start_dim=-2)  # B, N, Cv, 2L
        mem_out = torch.matmul(mv, p_aff)                # B, N, Cv, HW
        mem_out = mem_out.view(B, N, -1, H, W)             # B, N, Cv, H, W

        return S, mem_out

    def matching(self, qk, qv):
        # memory
        mk, mv = self.get_mem()                                # kappa, B, N, 2, Ck, L; val, B, N, 2, Cv, L
        # l2norm
        qk = l2norm(qk, dim=1)                                   # B, Ck, H, W
        mk = l2norm(mk, dim=-2)
        # affinity
        S, mem_out = self.get_affinity(qk, mk, mv)     # B, N, 2L, HW; BN, K*2, H, W; BN, 2, L, H, W
        qv = qv.unsqueeze(1).expand_as(mem_out)
        # flatten
        mem_out = mem_out.flatten(end_dim=1)                              # BN, Cv, H, W
        qv = qv.flatten(end_dim=1)                                  # BN, Cv, H, W
        # fusion all features as object context
        context_out = self.fusion_layer(torch.cat([mem_out, qv, S], dim=1))

        return context_out, mk.shape[1]

    def get_mem(self):
        # define memory
        kappas = []
        nus = []
        for key, mem in self.memories.items():
            if mem.bases is not None:
                kappas.append(mem.bases['kappa'])
                nus.append(mem.bases['nu'])

        kappa = torch.cat(kappas, dim=-1)    # B, N, 2, Ck, 2L
        nu = torch.cat(nus, dim=-1)          # B, N, 2, Cv, 2L
        return kappa, nu

    def forward(self, qk, qv):
        pass

