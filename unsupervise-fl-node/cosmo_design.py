from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(0)

fusion_weight = np.array([[0.29944117, 0.56794159, 0.13261724],
       [0.02646727, 0.23987183, 0.7336609 ],
       [0.43845882, 0.49335323, 0.06818795],
       [0.35973997, 0.56481998, 0.07544006],
       [0.22159766, 0.48902021, 0.28938213],
       [0.19107792, 0.47045704, 0.33846504],
       [0.44423391, 0.54002756, 0.01573852],
       [0.30068799, 0.38804443, 0.31126758],
       [0.46167461, 0.21458298, 0.32374241]])

## audio, depth, radar
def FeatureConstructor(f1, f2, f3, num_positive):

    # fusion_weight = np.arange(1, num_positive + 1) / (num_positive + 1)#(0.1, 0,2, ..., 0.9)
    fusion_weight = np.random.dirichlet((1, 1, 1), num_positive)
    # print("fusion_weight: ",fusion_weight)

    fused_feature = []

    for fuse_id in range(num_positive):

        temp_fuse = fusion_weight[fuse_id, 0] * f1 + fusion_weight[fuse_id, 1] * f2 + fusion_weight[fuse_id, 2] * f3
        # temp_fuse = torch.cat((fusion_weight[fuse_id, 0] * f1, fusion_weight[fuse_id, 1] * f2, fusion_weight[fuse_id, 2] * f3), dim=1) #concate

        fused_feature.append(temp_fuse)

    fused_feature = torch.stack(fused_feature, dim = 1)

    return fused_feature


## contrastive loss with supervised format

class CosmoLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CosmoLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)# change to [n_views*bsz, 3168]
        contrast_feature = F.normalize(contrast_feature, dim = 1)

        # print(contrast_feature.shape)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits, z_i * z_a / T
        similarity_matrix = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        # similarity_matrix = F.normalize(similarity_matrix, p=2, dim = 1)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)# positive index
        # print(mask.shape)#[1151, 1152] (btz*9)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )#dig to 0, others to 1 (negative samples)
        # print(logits_mask.shape)

        mask = mask * logits_mask#positive samples except itself

        # compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask #exp(z_i * z_a / T)
        # all_log_prob = torch.log(exp_logits.sum(1))# log(sum(exp(z_i * z_a / T))), need change to I\{i} later

        # SupCon out
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)#sup_out

        # SupCon in
        # log_prob =  torch.exp(similarity_matrix) / exp_logits.sum(1, keepdim=True)
        # mean_log_prob_pos = torch.log((mask * log_prob).sum(1) / mask.sum(1))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print("loss:",loss)

        loss = loss.view(anchor_count, batch_size).mean()
        # print("mean loss:",loss)

        return loss
