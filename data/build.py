# encoding: utf-8
"""
@author:  Minye Wu
@GITHUB: wuminye
"""

from torch.utils import data

from .datasets.ray_source import IBRay_NHR, IBRay_NHR_View
from .transforms import build_transforms


def build_dataset(data_folder_path,  transforms, bunch):
    datasets = IBRay_NHR(data_folder_path, transforms=transforms, bunch=bunch)
    return datasets

def build_dataset_view(data_folder_path,  transforms):
    datasets = IBRay_NHR_View(data_folder_path, transforms=transforms)
    return datasets


def make_data_loader(cfg, is_train=True):

    batch_size = cfg.SOLVER.IMS_PER_BATCH

    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg.DATASETS.TRAIN, transforms, bunch=cfg.SOLVER.BUNCH)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader, datasets


def make_data_loader_view(cfg, is_train=False):

    batch_size = cfg.SOLVER.IMS_PER_BATCH


    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset_view(cfg.DATASETS.TRAIN, transforms)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader, datasets
