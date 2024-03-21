""" Dataset loader for the Charades-STA dataset """
import os
import json
from .BaseDataset import BaseDataset

class Charades(BaseDataset):
    def __init__(self, split):
        # statistics for all video length
        # min:12 max:390 mean: 62, std:18
        # max sentence length：train->10, test->10
        super(Charades, self).__init__(split)

    def __len__(self):
        return len(self.annotations)

    def get_annotation(self):

        with open(
                os.path.join(self.anno_dirs['Charades'],
                             '{}_annotation.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)


        return annotations