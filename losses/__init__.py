from collections import defaultdict

from .bce_losses import *
from .lovasz_losses import *


loss_map = {
    'ce': CE,
    'boots_ce': BootstrappedCE,
    'iou': mask_iou_loss,
    'lovasz': lovasz_softmax
}


class VOSLoss(nn.Module):
    def __init__(self, config_loss, max_iter, device):
        super(VOSLoss, self).__init__()
        assert (config_loss.NAME in loss_map.keys()) and (config_loss.AUX is None or config_loss.AUX in loss_map.keys())
        assert (max_iter > 0)

        start_warm = config_loss.BS_PERIOD[0]  # * max_iter
        end_warm = config_loss.BS_PERIOD[1]    # * max_iter 

        top_p = config_loss.BS_RATIO
        self.main_loss = loss_map[config_loss.NAME](start_warm, end_warm, top_p).to(device)

        if config_loss.AUX is not None:
            self.aux_loss = loss_map[config_loss.AUX]
        else:
            self.aux_loss = lambda x, y: torch.zeros([1]).to(device)

        self.aux_alpha = config_loss.AUX_RATIO

    def forward(self, scores, target, it, valid_obj=None):
        """
        :param scores:  B x (N+1) x (T-1) x H x W), scores
        :param target: B x (T-1) x H x W
        :param valid_obj: valid flag for each obj, B x (N+1), value is 0 or 1
        :param it: current iterations
        :return: mean loss and percentage (1.0 here)
        """
        losses = defaultdict(int)
        loss_main, p = self.main_loss(scores, target, it, valid_obj)
        B, N, T, H, W = scores.shape
        if valid_obj is None:
            pred = F.softmax(scores.transpose(1, 2), dim=2).view(B*T, N, H, W)
            target_ = target.contiguous().view(B*T, H, W)
            loss_aux = self.aux_loss(pred, target_)
        else:
            loss_aux = 0.0
            for b in range(B):
                cur_scores = scores[b][valid_obj[b] > 0.5]                # N, T, H, W
                cur_pred = F.softmax(cur_scores.transpose(0, 1), dim=1)   # T x N x H x W
                loss_aux += self.aux_loss(cur_pred, target[b])

            loss_aux = loss_aux / B

        losses['total_loss'] = loss_main + self.aux_alpha * loss_aux
        losses['main_loss'] = loss_main
        losses['aux_loss'] = loss_aux
        losses['p'] = p

        return losses


def get_criterion(config_loss, logger, rank, max_iter, device):
    if rank <= 0:
        logger.info(f'OnlyValidObject: {config_loss.ONLY_VALID_OBJ}, Main Loss: {config_loss.NAME}, '
                    f'p: {config_loss.BS_RATIO}, Aux Loss: {config_loss.AUX}, '
                    f'the ratio of Aux Loss: {config_loss.AUX_RATIO}')
    return VOSLoss(config_loss, max_iter, device)
