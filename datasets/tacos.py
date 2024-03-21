""" Dataset loader for the TACoS dataset """
import os
import json

from .BaseDataset import BaseDataset


class TACoS(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:90 max:2578 mean: 528, std:436
        # max sentence length：train-->46, test-->50
        super(TACoS, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):
        with open(
                os.path.join(self.anno_dirs['TACoS'],
                             '{}_annotation.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)

        return annotations