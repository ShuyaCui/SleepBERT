from abc import *
import random


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.sample_count = args.sample_count
        self.bin_num = args.bin_num
        # self.save_folder = dataset._get_preprocessed_folder_path()
        # 调用AbstractDataset类中的load_dataset函数
        # 得到处理后的train, val, test
        dataset, bin_edges = dataset.load_dataset()
        self.bin_edges = bin_edges
        # 分别取出对应的内容
        self.data_train = dataset['train']
        if self.args.val == True:
           self.data_val = dataset['val']
        if self.args.test == True:
           self.data_test = dataset['test']

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass



