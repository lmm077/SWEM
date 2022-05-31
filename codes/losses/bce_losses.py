import torch
import torch.nn as nn
import torch.nn.functional as F


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, scores, target, it, valid_obj=None):
        """
        :param scores:  B x (N+1) x (T-1) x H x W)
        :param target: B x (T-1) x H x W
        :param valid_obj: valid flag for each obj, B x (N+1), value is 0 or 1
        :param it: current iterations
        :return: mean loss and percentage
        """
        B, T, H, W = target.shape
        if valid_obj is not None:
            if it < self.start_warm:
                mean_loss = 0.0
                for b in range(B):
                    #assert (valid_obj[b].sum() == target[b].max()+1)
                    cur_scores = scores[b][valid_obj[b] > 0.5].unsqueeze(0)
                    mean_loss += F.cross_entropy(cur_scores, target[b].unsqueeze(0))
                return mean_loss / B, 1.0
            raw_losses = []
            for b in range(B):
                cur_scores = scores[b][valid_obj[b] > 0.5].unsqueeze(0)
                raw_losses.append(F.cross_entropy(cur_scores, target[b].unsqueeze(0), reduction='none').view(1, T, -1))

            raw_loss = torch.cat(raw_losses, dim=0)                                        # B x T x (H*W)
        else:
            if it < self.start_warm:
                return F.cross_entropy(scores, target), 1.0
            raw_loss = F.cross_entropy(scores, target, reduction='none').view(B, T, -1)     # B x T x (H*W)

        num_pixels = H * W

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, k=int(num_pixels * this_p), dim=-1, sorted=False)    # B, T, k
        return loss.mean(), this_p


class CE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, scores, target, it, valid_obj=None):
        """
        :param scores:  B x (N+1) x (T-1) x H x W)
        :param target: B x (T-1) x H x W
        :param valid_obj: valid flag for each obj, B x (N+1), value is 0 or 1
        :param it: current iterations
        :return: mean loss and percentage (1.0 here)
        """
        if valid_obj is not None:
            B, T, H, W = target.shape
            mean_loss = 0.0
            for b in range(B):
                cur_scores = scores[b][valid_obj[b] > 0.5].unsqueeze(0)
                mean_loss += F.cross_entropy(cur_scores, target[b].unsqueeze(0))
            return mean_loss / B, 1.0
        else:
            return F.cross_entropy(scores, target), 1.0


def binary_entropy_loss(pred, target, num_object, eps=0.001):

    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)

    loss = torch.mean(ce)

    # TODO: training with bootstrapping

    return loss

def cross_entropy_loss(pred, mask, bootstrap=0.4):

    # pred: [N x T x (nobjs+1) x H x W] soft prediction
    # mask: [N x T x (nobjs+1) x H x W] one-hot encoded
    N, T, nobjs_1, H, W = mask.shape

    pred = -1 * torch.log(pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)

    loss = torch.sum(pred * mask, dim=2).view(N, T, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :, :num])

    return loss

def mask_iou(pred, target):

    """
    param: pred of size [B x N x H x W]
    param: target of size [B x N x H x W]
    """

    assert len(pred.shape) == 4 and pred.shape == target.shape

    B, N = pred.shape[:2]

    inter = torch.min(pred, target).sum(dim=(-1, -2))
    union = torch.max(pred, target).sum(dim=(-1, -2)) + 1e-6

    iou = torch.sum(inter / union) / (B*N)
    #print(iou)
    return iou

def mask_iou_loss(pred, label):
    # pred, (B*T) x N x H x W
    # label, (B*T) x H x W
    B, N = pred.shape[:2]
    target = torch.zeros_like(pred)
    for n in range(N):
        target[:, n] = (label == n).type(pred.dtype)

    loss = 1.0 - mask_iou(pred, target)
    return loss





