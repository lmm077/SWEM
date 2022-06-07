import torch
from torch import nn
from torch.nn import functional as F

from methods.basic_modules.networks import KeyEncoder, ValueEncoderSO, ValueEncoder, KeyProjection, Decoder
from .modules import SWEMCore

# Weighted Expectation-Maximization Network
class SWEM(nn.Module):
    def __init__(self, config_model):
        super(SWEM, self).__init__()
        # hyper-parameters
        keydim = config_model.KEYDIM
        valdim = config_model.VALDIM
        n_bases = config_model.NUM_BASES
        n_iters = config_model.NUM_EM_ITERS
        tau = config_model.EM_TAU
        topl = config_model.TOPL

        self.single_object = config_model.SINGLE_OBJ

        # module construction
        self.key_encoder = KeyEncoder(config_model.BACKBONE)
        if self.single_object:
            self.value_encoder = ValueEncoderSO(self.key_encoder.num_features[0])
        else:
            self.value_encoder = ValueEncoder(self.key_encoder.num_features[0])

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(self.key_encoder.num_features[0], keydim=keydim)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(self.key_encoder.num_features[0], valdim, kernel_size=(3, 3), padding=(1, 1))

        self.swem_core = SWEMCore(n_bases=n_bases, valdim=valdim, n_iters=n_iters, tau=tau, topl=topl)

        self.decoder = Decoder([valdim, self.key_encoder.num_features[1], self.key_encoder.num_features[2]], 256)

    def encode_key(self, frames):
        s16, s8, s4 = self.key_encoder(frames)
        qk16 = self.key_proj(s16)
        qv16 = self.key_comp(s16)
        return qk16, qv16, s16, s8, s4

    def encode_value(self, frame, masks, s16):
        # masks, B, N+1, H4, W4
        # frame, B, C, H, W
        N = masks.shape[1] - 1
        other_masks = (1 - masks - masks[:, 0:1])
        mask_fg = masks[:, 1:].flatten(end_dim=1).unsqueeze(1)                     # BN, 1, H, W
        mask_ot = other_masks[:, 1:].flatten(end_dim=1).unsqueeze(1)
        frame = frame.unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(end_dim=1)   # BN, C, H, W
        s16 = s16.unsqueeze(1).expand(-1, N, -1, -1, -1).flatten(end_dim=1)       # BN, Cs, H16, W16

        if self.single_object:
            mv16 = self.value_encoder(frame, s16, mask_fg)
        else:
            mv16 = self.value_encoder(frame, s16, mask_fg, mask_ot)

        mv16 = mv16.view(-1, N, *mv16.shape[1:])

        return mv16

    def init_mem(self, qk16, mv16, mask):
        self.swem_core.empty()
        mem = self.memorize(qk16, mv16, mask, mask.float())
        return mem

    def memorize(self, qk16, mv16, masks_hard, masks_soft):
        """
        :param k16: Key features, [b, c_k16, h_16, w_16]
        :param v16: Value features, [b, c_v16, h_16, w_16]
        :param masks_hard: [b, nobjs+1, h_m, w_m]
        :param masks_soft: [b, nobjs+1, h_m, w_m]
        :return: hard_samples, mems, mems_hard
        """
        b, c_k16, h_16, w_16 = qk16.shape

        # process masks
        masks_hard_ = F.interpolate(masks_hard[:, 1:].float(), size=(h_16, w_16), mode='nearest')
        masks_soft_ = F.interpolate(masks_soft[:, 1:], size=(h_16, w_16), mode='bilinear')
        masks_fg = masks_hard_ * masks_soft_
        masks_bg = (1 - masks_hard_) * (1 - masks_soft_)
        masks = torch.stack([masks_bg, masks_fg], dim=2)  # B, N, 2, H, W

        self.swem_core.memorize(qk16, mv16, masks)

    def match(self, qk16, qv16):
        context_out, nobjs = self.swem_core.matching(qk16, qv16)
        return context_out, nobjs

    def decode(self, n, context, s8, s4, valid_obj, out_size):
        # expand skip features for multiple objects
        s8_expand = s8.unsqueeze(1).expand(-1, n, -1, -1, -1).flatten(end_dim=1)
        s4_expand = s4.unsqueeze(1).expand(-1, n, -1, -1, -1).flatten(end_dim=1)

        # decoding
        logits = self.decoder(context, s8_expand, s4_expand, out_size)  # BN, 1, H_o, W_o
        preds = torch.sigmoid(logits)
        preds = preds.view(-1, n, *preds.shape[-2:])                    # B, N, H_o, W_o

        if valid_obj is not None:
            preds = preds * valid_obj[:, 1:].unsqueeze(2).unsqueeze(2)
        logits = self.aggregate(preds)                                  # B, N+1, H_o, W_o

        pred_mask = F.softmax(logits, dim=1)

        return logits, pred_mask

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'init':
            return self.init_mem(*args, **kwargs)
        elif mode == 'memorize':
            return self.memorize(*args, **kwargs)
        elif mode == 'match':
            return self.match(*args, **kwargs)
        elif mode == 'segment':
            return self.decode(*args, **kwargs)
        else:
            raise NotImplementedError

