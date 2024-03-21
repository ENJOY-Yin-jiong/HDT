import os
import json
import h5py

from .BaseDataset import BaseDataset

class DiDeMo(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:12 max:390 mean: 62, std:18
        # max sentence length：train->10, test->10
        super(DiDeMo, self).__init__(split)
        self.videos = h5py.File(os.path.join(self.feature_dirs['DiDeMo'], 'didemo_5fps_vgg16_trimmed_original.hdf5'))

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):

        with open(
                os.path.join(self.anno_dirs['DiDeMo'],
                             '{}_annotation.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)

        return annotations