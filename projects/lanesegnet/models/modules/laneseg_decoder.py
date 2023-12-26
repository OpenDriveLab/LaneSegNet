#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LaneSegNetDecoder(TransformerLayerSequence):

    def __init__(self, 
                 *args, 
                 return_intermediate=False, 
                 pc_range=None, 
                 sample_idx=[0, 3, 6, 9], # num_ref_pts = len(sample_idx) * 2
                 pts_dim=3, 
                 **kwargs):
        super(LaneSegNetDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.pc_range = pc_range
        self.sample_idx = sample_idx
        self.pts_dim = pts_dim

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):

        output = query
        intermediate = []
        intermediate_reference_points = []
        lane_ref_points = reference_points[:, :, self.sample_idx * 2, :]
        for lid, layer in enumerate(self.layers):
            # BS NUM_QUERY NUM_LEVEL NUM_REFPTS 3
            reference_points_input = lane_ref_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                reg_center = reg_branches[0]
                reg_offset = reg_branches[1]

                tmp = reg_center[lid](output)
                bs, num_query, _ = tmp.shape
                tmp = tmp.view(bs, num_query, -1, self.pts_dim)

                assert reference_points.shape[-1] == self.pts_dim

                tmp = tmp + inverse_sigmoid(reference_points)
                tmp = tmp.sigmoid()
                reference_points = tmp.detach()

                coord = tmp.clone()
                coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                if self.pts_dim == 3:
                    coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                centerline = coord.view(bs, num_query, -1).contiguous()

                offset = reg_offset[lid](output)
                left_laneline = centerline + offset
                right_laneline = centerline - offset
                left_laneline = left_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]
                right_laneline = right_laneline.view(bs, num_query, -1, self.pts_dim)[:, :, self.sample_idx, :]

                lane_ref_points = torch.cat([left_laneline, right_laneline], axis=-2).contiguous().detach()

                lane_ref_points[..., 0] = (lane_ref_points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                lane_ref_points[..., 1] = (lane_ref_points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                if self.pts_dim == 3:
                    lane_ref_points[..., 2] = (lane_ref_points[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER.register_module()
class CustomDetrTransformerDecoderLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(CustomDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
