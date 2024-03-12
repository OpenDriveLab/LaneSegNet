#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class RelationshipHead(BaseModule):

    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=True,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25),
                 init_cfg=None):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)
        self.classifier = MLP(256, 256, 1, 3)
        self.loss_rel = build_loss(loss_rel)

    def forward_train(self, o1_feats, o1_assign_results, o2_feats, o2_assign_results, gt_adj):
        rel_pred = self.forward(o1_feats, o2_feats)
        losses = self.loss(rel_pred, gt_adj, o1_assign_results, o2_assign_results)
        return losses

    def get_relationship(self, o1_feats, o2_feats):
        rel_pred = self.forward(o1_feats, o2_feats)
        rel_results = rel_pred.squeeze(-1).sigmoid()
        rel_results = [_ for _ in rel_results]
        return rel_results

    def forward(self, o1_feats, o2_feats):
        # feats: D, B, num_query, num_embedding
        o1_embeds = self.MLP_o1(o1_feats[-1])
        o2_embeds = self.MLP_o2(o2_feats[-1])

        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)
        relationship_pred = self.classifier(relationship_tensor)

        return relationship_pred

    def loss(self, rel_preds, gt_adjs, o1_assign_results, o2_assign_results):
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results[-1]
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results[-1]
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        targets = []
        masked_rel_preds = []
        for i in range(B):
            gt_adj = gt_adjs[i]
            len_o1 = gt_adj.size(0)
            len_o2 = gt_adj.size(1)
            o1_pos_mask = o1_pos_assigned_gt_inds[i] < len_o1
            o2_pos_mask = o2_pos_assigned_gt_inds[i] < len_o2

            masked_rel_pred = rel_preds[i][o1_pos_inds[i]][:, o2_pos_inds[i]][o1_pos_mask][:, o2_pos_mask]
            masked_rel_preds.append(masked_rel_pred.view(-1, 1))

            target = gt_adj[o1_pos_assigned_gt_inds[i][o1_pos_mask]][:, o2_pos_assigned_gt_inds[i][o2_pos_mask]]
            targets.append(1 - target.view(-1).long())

        targets = torch.cat(targets, dim=0)
        rel_preds = torch.cat(masked_rel_preds, dim=0)
        if torch.numel(targets) == 0:
            return dict(loss_rel=rel_preds.sum() * 0)

        loss_rel = self.loss_rel(rel_preds, targets, avg_factor=targets.sum())

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_rel = torch.nan_to_num(loss_rel)

        return dict(loss_rel=loss_rel)
