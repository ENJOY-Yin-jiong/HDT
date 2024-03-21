import torch
import torch.nn.functional as F
from torch import nn


from .operation import mask_logits

def aligment_score(query_features,
                   video_features,
                   query_mask,
                   video_mask,
                   inner_label,
                   GT_inner=True):
    B, T, channels = video_features.shape

    # ABLATION ---- ACTION LOSS
    if query_features.dim() == 3:
        query_features = query_features.sum(1) / query_mask.sum(1).unsqueeze(1)
        query_features = F.normalize(query_features, p=2, dim=1)  # B, channels

    if GT_inner:
        frame_weights = inner_label / video_mask.sum(1, keepdim=True)
    else:
        norm_video = F.normalize(video_features, p=2, dim=-1)
        frame_weights = torch.bmm(query_features.unsqueeze(1),
                                  norm_video.transpose(1, 2))  # B,1,T
        frame_weights = mask_logits(frame_weights.squeeze(1),
                                    video_mask)  # B,T
        frame_weights = torch.softmax(frame_weights, dim=-1)

    # norm_video = F.normalize(video_features, p=2, dim=-1).contiguous()
    # # distance = torch.bmm(norm_video, norm_video.transpose(1, 2))
    # distance = torch.cdist(norm_video, norm_video)
    # # distance = 1 - distance
    # distance = distance * inner_label.unsqueeze(1) * inner_label.unsqueeze(
    #     2)
    # distance = torch.sum(distance.reshape(B, -1), dim=-1) / (
    #     torch.sum(inner_label, dim=1) * torch.sum(inner_label, dim=1) +
    #     1e-30)
    # distance = distance.mean()

    video_features = video_features * frame_weights.unsqueeze(2)
    video_features = video_features.sum(1)
    video_features = F.normalize(video_features, p=2, dim=1)
    # print(video_features.shape)
    # print(video_features.T.shape)
    video_sim = torch.matmul(video_features, video_features.T)
    video_sim = torch.softmax(video_sim, dim=-1)
    # print(query_features.shape)
    # print(query_features.T.shape)
    query_sim = torch.matmul(query_features, query_features.T)
    query_sim = torch.softmax(query_sim, dim=-1)
    kl_loss = (F.kl_div(query_sim.log(), video_sim, reduction='sum') +
               F.kl_div(video_sim.log(), query_sim, reduction='sum')) / 2

    # triplet loss to enforce query feature close to referenced video features
    # features = torch.cat((query_features, video_features), dim=0)
    # labels = torch.cat((torch.arange(B), torch.arange(B)), dim=0)
    # labels = labels.to(query_features.device).detach()
    # triplet_loss, _ = batch_all_triplet_loss(labels,
    #                                          features,
    #                                          margin=0.2,
    #                                          squared=True)

    # print(kl_loss, triplet_loss, distance)
    # return kl_loss * 100 + triplet_loss * 10 + distance * 0.1
    # return kl_loss, triplet_loss, distance
    return kl_loss

def L1_aligment_score(query_features,
                   video_features,
                   query_mask,
                   video_mask,
                   inner_label,
                   GT_inner=True):
    B, T, channels = video_features.shape

    l1_loss = nn.L1Loss()

    # ABLATION ---- ACTION LOSS
    if query_features.dim() == 3:
        query_features = query_features.sum(1) / query_mask.sum(1).unsqueeze(1)
        query_features = F.normalize(query_features, p=2, dim=1)  # B, channels

    if GT_inner:
        frame_weights = inner_label / video_mask.sum(1, keepdim=True)
    else:
        norm_video = F.normalize(video_features, p=2, dim=-1)
        frame_weights = torch.bmm(query_features.unsqueeze(1),
                                  norm_video.transpose(1, 2))  # B,1,T
        frame_weights = mask_logits(frame_weights.squeeze(1),
                                    video_mask)  # B,T
        frame_weights = torch.softmax(frame_weights, dim=-1)

    # norm_video = F.normalize(video_features, p=2, dim=-1).contiguous()
    # # distance = torch.bmm(norm_video, norm_video.transpose(1, 2))
    # distance = torch.cdist(norm_video, norm_video)
    # # distance = 1 - distance
    # distance = distance * inner_label.unsqueeze(1) * inner_label.unsqueeze(
    #     2)
    # distance = torch.sum(distance.reshape(B, -1), dim=-1) / (
    #     torch.sum(inner_label, dim=1) * torch.sum(inner_label, dim=1) +
    #     1e-30)
    # distance = distance.mean()

    video_features = video_features * frame_weights.unsqueeze(2)
    video_features = video_features.sum(1)
    video_features = F.normalize(video_features, p=2, dim=1)
    # print(video_features.shape)
    # print(video_features.T.shape)
    video_sim = torch.matmul(video_features, video_features.T)
    video_sim = torch.softmax(video_sim, dim=-1)
    # print(query_features.shape)
    # print(query_features.T.shape)
    query_sim = torch.matmul(query_features, query_features.T)
    query_sim = torch.softmax(query_sim, dim=-1)
    # kl_loss = (F.kl_div(query_sim.log(), video_sim, reduction='sum') +
               # F.kl_div(video_sim.log(), query_sim, reduction='sum')) / 2

    loss = l1_loss(query_sim, video_sim)

    # triplet loss to enforce query feature close to referenced video features
    # features = torch.cat((query_features, video_features), dim=0)
    # labels = torch.cat((torch.arange(B), torch.arange(B)), dim=0)
    # labels = labels.to(query_features.device).detach()
    # triplet_loss, _ = batch_all_triplet_loss(labels,
    #                                          features,
    #                                          margin=0.2,
    #                                          squared=True)

    # print(kl_loss, triplet_loss, distance)
    # return kl_loss * 100 + triplet_loss * 10 + distance * 0.1
    # return kl_loss, triplet_loss, distance
    return loss


def compute_loss(pred_start, pred_end, pred_inter, start_labels,
                 end_labels, inter_label, mask):

    # start_regularity = self.regularization_score(pred_start)
    # end_regularity = self.regularization_score(pred_end)

    start_loss = compute_boundary_loss(pred_start, start_labels)
    end_loss = compute_boundary_loss(pred_end, end_labels)
    inter_loss = compute_location_loss(pred_inter, inter_label, mask)

    return start_loss + end_loss, inter_loss


def compute_boundary_loss(pred, targets):
    return F.cross_entropy(pred, targets.long())
    # return self.regression_loss(pred, targets)


def compute_location_loss(pred, targets, mask):
    weights_per_location = torch.where(targets == 0.0, targets + 1.0,
                                       2.0 * targets)
    loss_per_location = nn.BCEWithLogitsLoss(reduction='none')(pred,
                                                               targets)
    loss_per_location = loss_per_location * weights_per_location
    mask = mask.type(torch.float32)
    loss = torch.sum(loss_per_location * mask,
                     dim=1) / (torch.sum(mask, dim=1) + 1e-13)
    return loss.mean()


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negtive samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg



def get_targets(self, boundary, num_clips):
    # print('unit:',self.unit)
    batch_size = boundary.size(0)
    avg_factor = 0

    center_tgt = boundary.new_zeros(batch_size, num_clips)
    window_tgt = boundary.new_zeros(batch_size, num_clips)
    offset_tgt = boundary.new_zeros(batch_size, num_clips)
    weight = boundary.new_zeros(batch_size, num_clips)

    for batch_id in range(batch_size):
        batch_boundary = boundary[batch_id]
        # printb('batch_boundary', atch_boundary)
        batch_boundary[:, 1] -= self.unit

        keep = batch_boundary[:, 0] != -1
        batch_boundary = batch_boundary[keep] / self.unit
        num_centers = batch_boundary.size(0)
        # print(num_centers)
        avg_factor += num_centers
        # print(avg_factor)

        centers = batch_boundary.mean(dim=-1).clamp(max=num_clips - 0.5)
        windows = batch_boundary[:, 1] - batch_boundary[:, 0]
        # print('centers:', centers)
        # print('windows', windows)

        for i, center in enumerate(centers):
            radius = (windows[i] * self.radius_factor).int().item()
            # print('radius', radius)
            # print('radius', radius)
            sigma = (radius + 1) * self.sigma_factor
            center_int = center.int().item()
            # print('center_int', center_int)

            heatmap = batch_boundary.new_zeros(num_clips)
            start = max(0, center_int - radius)
            end = min(center_int + radius + 1, num_clips)
            # print('start:', start, '   end:', end)

            kernel = torch.arange(start - center_int, end - center_int)
            kernel = (-kernel ** 2 / (2 * sigma ** 2)).exp()
            heatmap[start:end] = kernel

            center_tgt[batch_id] = torch.max(center_tgt[batch_id], heatmap)
            window_tgt[batch_id, center_int] = windows[i]
            # print('center_int:', center_int)
            # print('windows_iter_{i}:', windows[i])
            offset_tgt[batch_id, center_int] = center - center_int
            weight[batch_id, center_int] = 1
            # print(window_tgt)

    return center_tgt, window_tgt, offset_tgt, weight, avg_factor


def get_boundary(self, center_pred, window_pred, offset_pred):
    pad = (self.kernel - 1) // 2
    hmax = F.max_pool1d(center_pred, self.kernel, stride=1, padding=pad)
    keep = (hmax == center_pred).float()
    center_pred = center_pred * keep

    topk = min(self.max_num_moments, center_pred.size(1))
    scores, inds = torch.topk(center_pred, topk)

    # print('center_pred', center_pred)
    center = inds + offset_pred.gather(1, inds).clamp(min=0)
    # print('center', center)
    window = window_pred.gather(1, inds).clamp(min=0)
    # print('window', window)

    boundry = center.unsqueeze(-1).repeat(1, 1, 2)
    boundry[:, :, 0] = center - window / 2
    boundry[:, :, 1] = center + window / 2
    boundry = boundry.clamp(min=0, max=center_pred.size(1) - 1) * self.unit
    boundry[:, :, 1] += self.unit

    boundary = torch.cat((boundry, scores.unsqueeze(-1)), dim=2)
    return boundary


def compute_location_loss2(center_pred, window_pred, offset_pred, targets, mask):

    # boundary = get_boundary(center_pred, window_pred, offset_pred)

    targets = get_targets(targets, mask.size(1))
    center_tgt, window_tgt, offset_tgt, weight, avg_factor = targets

    center_loss = 1.0 * GaussianFocalLoss()(center_pred, center_tgt)
    window_loss = 0.1 * nn.L1Loss()(window_pred, window_tgt)
    offset_loss = 1.0 * nn.L1Loss()(offset_pred, offset_tgt)

    return center_loss + window_loss + offset_loss

def early_pred_loss(video_features, pred, targets, mask):
    return compute_location_loss(pred, targets, mask)


def intra_constrast_loss(video_features, video_mask, inner_label):

    frame_weights = inner_label / video_mask.sum(1, keepdim=True)
    
    pos_video_features = video_features * frame_weights.unsqueeze(2)
    pos_video_features = pos_video_features.sum(1)
    pos_video_features = F.normalize(pos_video_features, p=2, dim=1)

    neg_video_features = video_features * (1 - frame_weights).unsqueeze(2)
    neg_video_features = neg_video_features.sum(1)
    neg_video_features = F.normalize(neg_video_features, p=2, dim=1)

    # constrast loss
    euclidean_distance = F.pairwise_distance(pos_video_features, neg_video_features)
    loss_contrastive = torch.mean(2.0 - F.relu(euclidean_distance))
    
    return loss_contrastive