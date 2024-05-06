import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from ..geotransformer.modules.loss import WeightedCircleLoss
from ..geotransformer.modules.ops.transformation import apply_transform
from ..geotransformer.modules.registration.metrics import isotropic_transform_error,evaluate_pose_Rt
from ..geotransformer.modules.ops.pairwise_distance import pairwise_distance


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']

        gt_pose_node_corr_indices = output_dict['gt_pose_node_corr_indices']
        gt_pose_node_corr_overlaps = output_dict['gt_pose_node_corr_overlaps']
        masks = torch.gt(gt_pose_node_corr_overlaps, self.acceptance_overlap)
        gt_pose_node_corr_indices = gt_pose_node_corr_indices[masks]
        gt_pose_ref_node_corr_indices = gt_pose_node_corr_indices[:, 0]
        gt_pose_src_node_corr_indices = gt_pose_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_pose_ref_node_corr_indices, gt_pose_src_node_corr_indices] = 1.0

        # masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        # gt_node_corr_indices = gt_node_corr_indices[masks]
        # gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        # gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        # gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        # gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()
        precision_inp = gt_node_corr_map[gt_node_corr_indices[:,0], gt_node_corr_indices[:,1]].mean()

        return precision,precision_inp,torch.tensor(len(gt_node_corr_indices)),torch.tensor(len(ref_node_corr_indices))

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        # transform = torch.tensor(transform,dtype=torch.float32)
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        if len(ref_corr_points) == 0 or len(src_corr_points) == 0:
            precision = torch.tensor(0.,dtype=torch.float32)
        else:
            corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
            precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()


        # return precision,precision_inp,src_corr_points.shape[0],src_corr_points_inp.shape[0]
        return precision,torch.tensor(0),src_corr_points.shape[0],torch.tensor(0)

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']
        ref_points = output_dict['ref_points']
        src_corr_points = output_dict['src_corr_points']
        ref_corr_points = output_dict['ref_corr_points']
        transform = torch.cat([transform, torch.tensor([[0, 0, 0, 1]]).cuda()], dim=0)


        src_pr = apply_transform(src_corr_points,transform)
        rre, rte = isotropic_transform_error(transform, est_transform)
        #chamer
        cham = chamfer_distance(src_pr[None,...].cuda(), ref_corr_points[None,...], batch_reduction=None)[0].cpu()

        # transform = torch.cat([transform, torch.tensor([[0, 0, 0, 1]]).cuda()], dim=0)
        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte,cham, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision,c_precision_inp,c_len_in,c_len_ou= self.evaluate_coarse(output_dict)
        f_precision,f_precision_inp,len_ou,len_in = self.evaluate_fine(output_dict, data_dict)
        rre, rte,cham, rmse, recall = self.evaluate_registration(output_dict, data_dict)
        len_ou = torch.tensor(len_ou).cuda()
        len_in = torch.tensor(len_in).cuda()
        return {
            'PIR': c_precision,
            'PIR_inp': c_precision_inp,
            'IR': f_precision,
            'IR_inp': f_precision_inp,
            'len_out':len_ou,
            'len_inp':len_in,
            'len_inp_c':c_len_in,
            'len_out_c':c_len_ou,
            'RRE': rre,
            'RTE': rte,
            'cham': cham,
            'RMSE': rmse,
            'RR': recall,
        }
