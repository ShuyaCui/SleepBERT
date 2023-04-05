from .base import AbstractDataset

import csv
import numpy as np
from datetime import date


class ukbDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'UKB'

    def load_act_df(self, file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the first row
            next(reader)
            act = []
            for row in reader:
                if row[1] == "NA":
                    act.append(0)
                else:
                    act.append(row[1])
        act = np.array(act)
        act = act.astype(np.float)
        return act


