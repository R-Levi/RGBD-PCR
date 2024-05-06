import os
import socket
import torch
import numpy as np
import pickle
from functools import partial

import pytorch3d
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .video_dataset import VideoDataset
from .video_dataset_scannet import VideoDatasetScannet
from ..geotransformer.modules.ops import grid_subsample, radius_search

# Define some important paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# get dataset root
RGBD_3D_ROOT = "/home/levi/Data_work/PCD/3dmatch_image"
GEO_3DMATCH_ROOT = ""
SCANNET_ROOT = "/home/levi/Data_work/workplace/scan_test_out"


def build_dataset(cfg, split, overfit=None):
    """
    Builds a dataset from the provided dataset configs.
    Configs can be seen is configs/config.py
    """
    if cfg.name == "scannet":
        root_path = SCANNET_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"data/scannet_{split}.pkl")
        data_dict = load_pickle(dict_path)
        dataset = VideoDatasetScannet(cfg, root_path, data_dict, split)

        # Reduce ScanNet validation size to allow for more frequent validation
        if split == "valid" or split == "test":
            dataset.instances = dataset.instances[::10]

    elif cfg.name == "3DMatch":
        root_path = RGBD_3D_ROOT
        dict_path = os.path.join(PROJECT_ROOT, f"data/3dmatch_{split}.pkl")
        data_dict = load_pickle(dict_path)
        dataset = VideoDataset(cfg, root_path, data_dict, split)

        # Reduce ScanNet validation size to allow for more frequent validation
        if split == "valid":
            dataset.instances = dataset.instances[::10]
    else:
        raise ValueError("Dataset name {} not recognized.".format(cfg.name))

    # Overfit only loads a single batch for easy debugging/sanity checks
    if overfit is not None:
        assert type(overfit) is int
        dataset.instances = dataset.instances[: cfg.batch_size] * overfit

    return dataset


def build_loader(cfg, split, overfit=None,neighbor_limits=None):
    """
    Builds the dataset loader (including getting the dataset).
    """
    dataset = build_dataset(cfg, split, overfit)
    shuffle = (split == "train")
    batch_size = cfg.batch_size

    if neighbor_limits is None:
        train_dataset = build_dataset(cfg, "train", overfit)
        neighbor_limits = calibrate_neighbors_stack_mode(
            train_dataset,
            geo_custom_collate,
            num_stages=cfg.num_stages,
            voxel_size=cfg.voxel_size,
            search_radius=cfg.init_radius,
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        pin_memory=False,
        collate_fn=partial(geo_custom_collate,
            num_stages=cfg.num_stages,
            voxel_size=cfg.voxel_size,
            search_radius=cfg.init_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=True),
        num_workers=cfg.num_workers,
    )

    return loader,neighbor_limits


def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    if lengths_list[-1][0] > 400 or lengths_list[-1][1] > 400:
        points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.pop(-1)
        lengths_list.pop(-1)
        points_list.append(points)
        lengths_list.append(lengths)

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2



    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }



def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits

def geo_custom_collate(data_dicts,num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True):
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)
    if batch_size == 1:
        feats = torch.cat(collated_dict.pop('ref_points_feat') + collated_dict.pop('src_points_feat'), dim=0)
        points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
        lengths = torch.LongTensor([points.shape[0] for points in points_list])
        points = torch.cat(points_list, dim=0)
    else:
        points_f_0 = collated_dict.pop('ref_points_feat')
        points_0 = collated_dict.pop('ref_points')
        points_f_1 = collated_dict.pop('src_points_feat')
        points_1 = collated_dict.pop('src_points')
        feats = []
        lengths = []
        points = []

        for pf0,pf1 in zip(points_f_0,points_f_1):
            feats.append(torch.cat((pf0,pf1), dim=0))
        for p0,p1 in zip(points_0,points_1):
            points_list = [p0,p1]
            length = torch.LongTensor([points.shape[0] for points in points_list])
            lengths.append(length)
            points.append(torch.cat(points_list,dim=0))

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        if batch_size == 1:
            input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
            collated_dict.update(input_dict)
        else:
            input_dict = {}
            for i in range(batch_size):
                one_batch = precompute_data_stack_mode(points[i], lengths[i], num_stages, voxel_size, search_radius,neighbor_limits)
                for k,v in one_batch.items():
                    if k not in input_dict:
                        input_dict[k] = []
                    input_dict[k].append(v)
            collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    # if collated_dict['lengths'][0][0]<5000 and collated_dict['lengths'][0][1]<5000:
    #     return None

    return collated_dict

def custom_collate(data):
    out = {}
    keys = data[0].keys()
    for key in keys:
        if "points_rgb_" in key:
            view_i = key.split("_")[-1]
            xyz = [ins[f"points_{view_i}"] for ins in data]
            rgb = [ins[f"points_rgb_{view_i}"] for ins in data]
            out[key] = pytorch3d.structures.Pointclouds(xyz, None, rgb)
        elif "points_" in key:
            out[key] = pytorch3d.structures.Pointclouds([ins[key] for ins in data])
        else:
            out[key] = default_collate([ins[key] for ins in data])

    return out


# load pickle
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
