import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
from .KPConvFPN import KPConvFPN
from .backbones import ResNetEncoder
from src.geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from src.geotransformer.modules.ops import point_to_node_partition, index_select,apply_transform,pairwise_distance
from src.geotransformer.modules.registration import get_node_correspondences
from src.geotransformer.modules.sinkhorn import LearnableLogOptimalTransport


class GeoRGBD(nn.Module):
    def __init__(self, cfg):
        super(GeoRGBD, self).__init__()

        # Define model parameters
        self.voxel_size = cfg.voxel_size
        self.gt_Rt = cfg.gt_Rt
        print("use_gt_pose:", self.gt_Rt)
        self.num_points_in_patch = cfg.num_points_in_patch
        self.matching_radius = cfg.ground_truth_matching_radius
        self.img_scale = cfg.img_scale

        ##define Geotransformer
        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )
        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )
        self.optimal_transport = LearnableLogOptimalTransport(cfg.num_sinkhorn_iterations)

        # useless model delete
        # if self.visual:
        self.image_encoder = ResNetEncoder(3, 32)
        self.mlp_project = nn.Sequential(
            nn.Linear(32, 32),
            # It's actually BatchNorm given that is Num_Points x Features
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

    def forward(self, data_dict):
        output_dict = {}
        # Downsample point clouds get data
        feats = data_dict['features'].detach()
        # rt_0 = data_dict['Rt_0']
        rt_1 = data_dict['transform']
        # ref-->0  src-->1
        ref_length_c = data_dict['lengths'][-1][0].item()
        # print(ref_length_c)
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        ##piror_data
        ref_pc_piror_inds = data_dict["ref_pc_piror_inds"]
        src_pc_piror_inds = data_dict["src_pc_piror_inds"]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points
        # output_dict['transform'] = data_dict['Rt_1']

        ##_c points match _f points###
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )
        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        ref_point_to_node_1, ref_node_masks_1, ref_node_knn_indices_1, ref_node_knn_masks_1 = point_to_node_partition(
            ref_points, ref_points_c, 256
        )
        src_point_to_node_1, src_node_masks_1, src_node_knn_indices_1, src_node_knn_masks_1 = point_to_node_partition(
            src_points, src_points_c, 256
        )

        if self.gt_Rt:
            gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
                ref_points_c,
                src_points_c,
                ref_node_knn_points,
                src_node_knn_points,
                rt_1,
                self.matching_radius,
                ref_masks=ref_node_masks,
                src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks,
                src_knn_masks=src_node_knn_masks,
            )
            if len(gt_node_corr_indices)==0:
                gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
                    ref_points_c,
                    src_points_c,
                    ref_node_knn_points,
                    src_node_knn_points,
                    rt_1,
                    1.5,
                    ref_masks=ref_node_masks,
                    src_masks=src_node_masks,
                    ref_knn_masks=ref_node_knn_masks,
                    src_knn_masks=src_node_knn_masks,
                )
            if len(gt_node_corr_indices)==0:
                print("record\n")
                print('index: ', data_dict['index'])
                print(f"{data_dict['sequence_id'], data_dict['ref_path']},{data_dict['src_path']}")
                print(
                    f"gt_node_corr_indices:{len(gt_node_corr_indices)},src_points_c:{len(src_points_c)},ref_points_c:{len(ref_points_c)}")
                print(f"ref_pc_piror_inds:{len(ref_pc_piror_inds)},src_pc_piror_inds:{len(src_pc_piror_inds)}")

            output_dict['gt_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps
            output_dict['gt_pose_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_pose_node_corr_overlaps'] = gt_node_corr_overlaps
        else:
            gt_pose_node_corr_indices, gt_pose_node_corr_overlaps = get_node_correspondences(
                ref_points_c,
                src_points_c,
                ref_node_knn_points,
                src_node_knn_points,
                rt_1,
                self.matching_radius,
                ref_masks=ref_node_masks,
                src_masks=src_node_masks,
                ref_knn_masks=ref_node_knn_masks,
                src_knn_masks=src_node_knn_masks,
            )
            if len(gt_pose_node_corr_indices) == 0:
                gt_pose_node_corr_indices, gt_pose_node_corr_overlaps = get_node_correspondences(
                    ref_points_c,
                    src_points_c,
                    ref_node_knn_points,
                    src_node_knn_points,
                    rt_1,
                    1.5,
                    ref_masks=ref_node_masks,
                    src_masks=src_node_masks,
                    ref_knn_masks=ref_node_knn_masks,
                    src_knn_masks=src_node_knn_masks,
                )
            output_dict['gt_pose_node_corr_indices'] = gt_pose_node_corr_indices
            output_dict['gt_pose_node_corr_overlaps'] = gt_pose_node_corr_overlaps

            corres_est = data_dict['3d_corres_est']
            if len(corres_est)==0:
                corres_est = np.stack((np.random.randint(0, ref_points_f.shape[0], 500),
                                        np.random.randint(0, src_points_f.shape[0], 500)), axis=-1)
                corres_est = torch.from_numpy(corres_est)
                corres_est = corres_est.cuda()
                print("random")

            ref_corres_inds = corres_est[:,0]
            src_corres_inds = corres_est[:,1]
            output_dict['ref_corres_inp'] = ref_points[ref_corres_inds]
            output_dict['src_corres_inp'] = src_points[src_corres_inds]
            ref_points_inds_c = ref_point_to_node_1[ref_corres_inds]
            src_points_inds_c = src_point_to_node_1[src_corres_inds]
            corr_np = np.stack((ref_points_inds_c.detach().cpu().numpy(), src_points_inds_c.detach().cpu().numpy()),
                               axis=-1)
            x = corr_np[:, 0] + corr_np[:, 1] * 1j
            corr_np = corr_np[np.unique(x, return_index=True)[1]]
            gt_node_corr_indices = torch.from_numpy(corr_np)
            ref_src_mask = torch.logical_and(ref_node_masks.unsqueeze(1), src_node_masks.unsqueeze(0))
            gt_node_corr_indices = gt_node_corr_indices[
                torch.where(ref_src_mask[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]] == True)]

            ref_knn_dists = torch.linalg.norm(ref_node_knn_points - ref_points_c.unsqueeze(1), dim=-1)  # (M, K)
            src_knn_dists = torch.linalg.norm(src_node_knn_points - src_points_c.unsqueeze(1), dim=-1)  # (N,k)
            dist_mat = torch.abs(
                ref_knn_dists.unsqueeze(1).unsqueeze(-1) - src_knn_dists.unsqueeze(0).unsqueeze(2))  # diff (m,n,k,k)
            dist_mat = dist_mat[gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]]  # c, k ,k
            dist_mat_mask = torch.logical_and(ref_node_knn_masks[gt_node_corr_indices[:, 0]].unsqueeze(-1),
                                              src_node_knn_masks[gt_node_corr_indices[:, 1]].unsqueeze(1))
            dist_mat.masked_fill_(~dist_mat_mask, 1e12)
            ratio = 0.05
            dist_mat_overlap_mask = torch.ones_like(dist_mat) * ratio

            dist_mat_overlap_mask = dist_mat <= dist_mat_overlap_mask

            ref_mat_overlap = torch.count_nonzero(dist_mat_overlap_mask, dim=-1)
            ref_mat_overlap = torch.sum(torch.gt(ref_mat_overlap, 0), dim=-1)
            ref_mat_overlap_ratio = ref_mat_overlap / ref_node_knn_masks.sum(-1)[gt_node_corr_indices[:, 0]]

            src_mat_overlap = torch.count_nonzero(dist_mat_overlap_mask, dim=1)
            src_mat_overlap = torch.sum(torch.gt(src_mat_overlap, 0), dim=-1)
            src_mat_overlap_ratio = src_mat_overlap / src_node_knn_masks.sum(-1)[gt_node_corr_indices[:, 1]]
            gt_node_corr_overlaps = (ref_mat_overlap_ratio + src_mat_overlap_ratio) / 2.

            output_dict['gt_node_corr_indices'] = gt_node_corr_indices
            output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

            del corres_est, dist_mat, x,corr_np

        src_overlapped_points_c_idx = src_point_to_node_1[src_pc_piror_inds]
        ref_overlapped_points_c_idx = ref_point_to_node_1[ref_pc_piror_inds]
        src_overlapped_points_c_idx, ref_overlapped_points_c_idx = torch.unique(
            src_overlapped_points_c_idx), torch.unique(ref_overlapped_points_c_idx)
        src_points_c_idx = torch.arange(src_points_c.shape[0]).to(src_overlapped_points_c_idx.device)
        ref_points_c_idx = torch.arange(ref_points_c.shape[0]).to(src_overlapped_points_c_idx.device)
        src_no_overlapped_points_c_list = [i.item() for i in src_points_c_idx if
                                           i not in src_overlapped_points_c_idx]
        src_no_overlapped_points_c_idx = torch.from_numpy(np.array(src_no_overlapped_points_c_list))
        ref_no_overlapped_points_c_list = [i.item() for i in ref_points_c_idx if
                                           i not in ref_overlapped_points_c_idx]
        ref_no_overlapped_points_c_idx = torch.from_numpy(np.array(ref_no_overlapped_points_c_list))

        # 2. KPFCNN Encoder
        feats_list,feats_s1 = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        ref_feats_c, src_feats_c, ref_embeddings, src_embeddings = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
            ref_overlapped_points_c_idx, src_overlapped_points_c_idx,
            ref_no_overlapped_points_c_idx, src_no_overlapped_points_c_idx
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )
            # output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            # output_dict['src_node_corr_indices'] = src_node_corr_indices

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats,
                                       src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform

        return output_dict