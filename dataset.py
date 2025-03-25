from torch.utils.data import Dataset, DataLoader
from torch import tensor
from datasets import load_dataset
from re import findall
from itertools import chain
from os import path
from typing import List, Tuple, Iterator, Dict
from tokenizer import Tokenizer
from PARAMETER import DATASET, DATASETS, TOKENIZER, TRAIN, MODEL


class Datasets(object):
    class _Dataset(Dataset):
        def __init__(self, dataset:List[int], end:int):
            self._dataset = dataset
            self._end = end
            self._datasetLen = len(self._dataset)

        def __getitem__(self, index:int) -> Tuple[int, int]:
            if index==self._datasetLen-len(self._end):
                return (tensor(self._dataset[index:index+MODEL.SEQ_LEN]), tensor(self._dataset[index+1:index+1+MODEL.SEQ_LEN-len(self._end)]+self._end))
            return (tensor(self._dataset[index:index+MODEL.SEQ_LEN]), tensor(self._dataset[index+1:index+1+MODEL.SEQ_LEN]))

        def __len__(self) -> int:
            return len(self._dataset)

    @staticmethod
    def makeDataset(config:DATASET) -> Dict[str, List[str]]:
        if not path.exists(config.DATASET):
            from huggingface_hub import login
            login(token=config.HG_TOKEN, add_to_git_credential=False)
        dataset_ = load_dataset(path=config.DATASET)["train"].train_test_split(test_size=config.TEST and 0.0005 or None, seed=123, shuffle=True)
        _ = {}
        tokenizer = Tokenizer(TOKENIZER)
        for dataset in dataset_:
            __ = []
            for data in dataset_[dataset]:
                __ += findall(r"\S+", data["input"]+data["reasoning_content"]+data["content"])
            __ = (list(chain(*tokenizer.encode(__))), tokenizer.encode("。"))
            _[dataset] = __
        return _

    def __init__(self, datasets:DATASETS):    
        self._datasets = map(self.makeDataset, datasets)
        self._train = True

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            datasetPair = next(self._datasets)
            datasetTrain, end = datasetPair.pop("train")
            self._trainDataset = {"dataset":datasetTrain, "end":end}
            datasetVal, end = datasetPair.pop("test")
            self._valDataset = {"dataset":datasetVal, "end":end}
            return True
        except:
            return False

    def __len__(self) -> int:
        if self._train:
            return len(self._trainDataset["dataset"])
        else:
            return len(self._valDataset["dataset"])

    def train(self):
        self._train = True

    def val(self):
        self._train = False

    def getData(self) -> Iterator:
        if self._train:
            return DataLoader(
                self._Dataset(**self._trainDataset), 
                batch_size=TRAIN.BATCH, 
                pin_memory=True, 
                pin_memory_device=TRAIN.DEVICE, 
                drop_last=True)
        else:
            return DataLoader(
                self._Dataset(**self._valDataset), 
                batch_size=TRAIN.BATCH, 
                pin_memory=True, 
                pin_memory_device=TRAIN.DEVICE, 
                drop_last=True)