""" Dataset loader for the Charades-STA dataset """
import os
import json
import torch
import h5py
import torch.nn.functional as F
import numpy as np

from .BaseDataset import BaseDataset
from core.config import config
from . import average_to_fixed_length

class CharadesVGG(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:12 max:390 mean: 62, std:18
        # max sentence length：train->10, test->10
        super(CharadesVGG, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):

        with open(
                os.path.join(self.anno_dirs['Charades'],
                             '{}_annotation.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)

        return annotations

    def get_video_features(self, vid, dataset):
        try:
            hdf5_file = h5py.File("/home/data3/yinjiong/dataset/charades/vgg_rgb_features.hdf5", 'r')
            features = torch.from_numpy(hdf5_file[vid][:]).float()
        except Exception as e:
            print(vid, e)
        if features.shape[0] > 1500:
            features = average_to_fixed_length(features, num_sample_clips=1500)
        # features = average_to_fixed_length(features)
        frame_rate = 3 if dataset != "Charades" else 1
        features = features[list(range(0, features.shape[0], frame_rate))]
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)

        # flip the input in time direction
        flip_in_time_direction = False  # use for start/end label flip
        if (
            self.split == "train"
            and config.DATASET.FLIP_TIME
            and np.random.random() < 0.5
        ):
            features = torch.flip(features, dims=[0])
            flip_in_time_direction = True

        length = features.shape[0]
        prefix, suffix = 0, 0
        # add a mean_feature in front of and end of the video to double the time length
        if (
            self.split == "train"
            and config.DATASET.EXTEND_TIME
            and np.random.random() < 0.7
        ):
            # mean_feature = torch.mean(features, dim=0)
            # extend_feature = mean_feature.unsqueeze(0).repeat((prefix, 1))  # add mean feature
            # extend_feature = torch.zeros((prefix, features.shape[1]))      # add zeros feature
            #  --->add another_features start<---
            index = np.random.randint(len(self.annotations))  # another_video
            video_id = self.annotations[index]["video"]
            while video_id == vid:
                index = np.random.randint(len(self.annotations))  # another_video
                video_id = self.annotations[index]["video"]
            featurePath = os.path.join(self.feature_dirs[dataset], video_id + ".npy")
            another_features = np.load(featurePath)
            another_features = np.squeeze(another_features)
            another_features = torch.from_numpy(another_features).float()
            if another_features.shape[0] > 1500:
                another_features = average_to_fixed_length(
                    another_features, num_sample_clips=1500
                )
            another_features = another_features[
                list(range(0, another_features.shape[0], frame_rate))
            ]
            prefix = round(np.random.random() * another_features.shape[0])
            extend_feature = another_features[:prefix]
            assert extend_feature.shape[0] == prefix
            #  --->add another_features end<---
            features = torch.cat([extend_feature, features], dim=0)
        vis_mask = torch.ones((features.shape[0], 1))

        return features, vis_mask, prefix, suffix, flip_in_time_direction