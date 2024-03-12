import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core import multi_apply, multi_apply, reduce_mean
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.core import (
    multi_apply,
    reduce_mean,
)
from shapely.geometry import LineString


@HEADS.register_module()
class MapTRHead(DETRHead):

    def __init__(
        self,
        *args,
        with_box_refine=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=200,
        bev_w=200,
        pc_range=None,
        num_vec=50,
        pts_dim=3,
        num_pts_per_vec=20,
        num_pts_per_gt_vec=20,
        dir_interval=1,
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=2.0),
        **kwargs,
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        assert pts_dim in (2, 3)
        self.pts_dim = pts_dim

        self.with_box_refine = with_box_refine
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 3
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        super(MapTRHead, self).__init__(*args, transformer=transformer, **kwargs)

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self.loss_dir = build_loss(loss_dir)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_cls_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        self.instance_embedding = nn.Embedding(
            self.num_vec, self.embed_dims * 2
        )
        self.pts_embedding = nn.Embedding(
            self.num_pts_per_vec, self.embed_dims * 2
        )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @force_fp32(apply_to=("mlvl_feats",))
    def forward(self, mlvl_feats, bev_feats, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        outputs = self.transformer(
            mlvl_feats,
            bev_feats,
            object_query_embeds,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=None,
            img_metas=img_metas
        )

        # bev_embed,
        hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](
                hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2)
            )
            tmp = self.reg_branches[lvl](hs[lvl])

            coord = tmp.clone()
            coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            if self.pts_dim == 3:
                coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            coord = coord.reshape(bs, self.num_vec, self.num_pts_per_vec, self.pts_dim)

            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            "all_cls_scores": outputs_classes,
            "all_pts_preds": outputs_pts_coords,
        }

        return outs

    def _get_target_single(
        self,
        cls_score,
        pts_pred,
        gt_shifts_pts,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        num_bboxes = pts_pred.size(0)
        # assigner and sampler
        gt_c = pts_pred.shape[-1]

        assign_result, order_index = self.assigner.assign(
            pts_pred,
            cls_score,
            gt_shifts_pts,
            gt_labels,
            gt_bboxes_ignore,
        )

        sampling_result = self.sampler.sample(assign_result, pts_pred, gt_shifts_pts)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_shifts_pts.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_shifts_pts.new_ones(num_bboxes)

        assigned_shift = order_index[pos_inds, pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros(
            (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
        )
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        pts_targets[pos_inds] = gt_shifts_pts[
            pos_assigned_gt_inds, assigned_shift, :, :
        ]
        return (
            labels,
            label_weights,
            pts_targets,
            pts_weights,
            pos_inds,
            neg_inds,
            pos_assigned_gt_inds,
        )

    def get_targets(
        self,
        cls_scores_list,
        pts_preds_list,
        gt_shifts_pts_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):

        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
            pos_assigned_gt_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            pts_preds_list,
            gt_shifts_pts_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list,
            neg_inds=neg_inds_list,
            pos_assigned_gt_inds=pos_assigned_gt_inds_list,
        )
        return (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
            assign_result,
        )

    def loss_single(
        self,
        cls_scores,
        pts_preds,
        gt_shifts_pts_list,
        gt_labels_list,
        gt_bboxes_ignore_list=None,
    ):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            pts_preds_list,
            gt_shifts_pts_list,
            gt_labels_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
            assign_result,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(
                pts_preds,
                size=(self.num_pts_per_gt_vec),
                mode="linear",
                align_corners=True,
            )
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()
        isnotnan = torch.isfinite(pts_preds).all(dim=-1).all(dim=-1)

        loss_pts = self.loss_bbox(
            pts_preds[isnotnan, :, :],
            pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )
        dir_weights = pts_weights[:, : -self.dir_interval, 0]
        denormed_pts_preds = pts_preds
        denormed_pts_preds_dir = (
            denormed_pts_preds[:, self.dir_interval :, :]
            - denormed_pts_preds[:, : -self.dir_interval, :]
        )
        pts_targets_dir = (
            pts_targets[:, self.dir_interval :, :]
            - pts_targets[:, : -self.dir_interval, :]
        )
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_pts, loss_dir, assign_result

    def shift_fixed_num_sampled_points(self, gt_areas_3d):
        """
        return [instances_num, num_shifts, fixed_num * self.pts_dim]
        """
        instances_list = []
        is_poly = False
        device = gt_areas_3d.device
        for fixed_num_pts in gt_areas_3d:
            # [fixed_num, pts_dim]
            fixed_num_pts = fixed_num_pts.reshape(-1, self.pts_dim)
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []

            if is_poly:
                fixed_num_pts = fixed_num_pts[:-1]
                for shift_right_i in range(fixed_num):
                    shift_pts = fixed_num_pts.roll(shift_right_i, 0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = torch.unsqueeze(pts_to_concat, axis=0)
                    shift_pts = torch.cat((shift_pts, pts_to_concat), axis=0)
                    shift_pts_list.append(shift_pts)
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if not is_poly:
                padding = torch.full(
                    [fixed_num - shift_pts.shape[0], fixed_num, self.pts_dim], -10000, device=device
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32, device=device)
        return instances_tensor

    @force_fp32(apply_to=("preds_dicts"))
    def loss(
        self,
        preds_dicts,
        gt_areas_3d,
        gt_labels_list,
        gt_bboxes_ignore=None,
        img_metas=None,
    ):

        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )
        gt_areas_3d = copy.deepcopy(gt_areas_3d)

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_pts_preds = preds_dicts["all_pts_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_shifts_pts_list = [
            self.shift_fixed_num_sampled_points(gt_area).to(device)
            for gt_area in gt_areas_3d
        ]

        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_pts, losses_dir, assign_result = (
            multi_apply(
                self.loss_single,
                all_cls_scores,
                all_pts_preds,
                all_gt_shifts_pts_list,
                all_gt_labels_list,
                all_gt_bboxes_ignore_list,
            )
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_reg"] = losses_pts[-1]
        loss_dict["loss_dir"] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1],
            losses_pts[:-1],
            losses_dir[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_dir"] = loss_dir_i
            num_dec_layer += 1
        return loss_dict, assign_result

    @force_fp32(apply_to=("preds_dicts"))
    def get_areas(self, preds_dicts, img_metas, rescale=False):

        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            pts = preds["areas"]
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([pts, scores, labels])
        return ret_list
