""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from .BaseDataset import BaseDataset
from . import average_to_fixed_length
from core.config import config


class ActivityNetCD(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:2 medium: max:1415  mean: 204, std:120
        # max sentence length：train-->73, test-->73
        super(ActivityNetCD, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):



        all_ids = os.listdir(self.feature_dirs['ActivityNet-CD'])
        all_ids = [s[:-4] for s in all_ids]  # remove ".npy" suffix

        # self.h5file = h5py.File(self.feature_dirs['ActivityNet-CD'], 'r')
        # all_ids = [key for key in self.h5file.keys()]

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json

        with open(
                os.path.join(self.anno_dirs['ActivityNet-CD'],
                             '{}_annotation.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        # elif self.split == "test_iid":
        #     with open(
        #             os.path.join(self.anno_dirs['ActivityNet-CD'],
        #                          '{}_annotation.json'.format(self.split)), 'r') as f:
        #         annotations = json.load(f)
        # elif self.split == "test_ood":
        #     with open(
        #             os.path.join(self.anno_dirs['ActivityNet-CD'],
        #                          '{}_annotation.json'.format(self.split)), 'r') as f:
        #         annotations = json.load(f)


        anno_pairs = []
        for anno in annotations:
            if not anno['video'] in all_ids:
                # print(vid, '{}.json'.format(self.split))
                continue
            else:
                anno_pairs.append(anno)


        return anno_pairs