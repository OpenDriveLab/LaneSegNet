#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from .lane_attention import LaneAttention


@TRANSFORMER.register_module()
class LaneSegNetTransformer(BaseModule):

    def __init__(self,
                 decoder=None,
                 embed_dims=256,
                 points_num=1,
                 pts_dim=3,
                 **kwargs):
        super(LaneSegNetTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.points_num = points_num
        self.pts_dim = pts_dim
        self.fp16_enabled = False
        self.init_layers()

    def init_layers(self):
        self.reference_points = nn.Linear(self.embed_dims, self.pts_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, LaneAttention):
                m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)


    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_embed,
                object_query_embed,
                bev_h,
                bev_w,
                reg_branches=None,
                cls_branches=None,
                **kwargs):

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)

        # ident init: repeat reference points to num points
        reference_points = reference_points.repeat(1, 1, self.points_num)
        reference_points = reference_points.sigmoid()
        bs, num_qeury, _ = reference_points.shape
        reference_points = reference_points.view(bs, num_qeury, self.points_num, self.pts_dim)

        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out
