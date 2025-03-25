from dataclasses import dataclass
from typing import List, Optional, Dict
from torch.accelerator import current_accelerator
from torch.accelerator import is_available as accelerator_is_available
from torch.cuda import is_available as cuda_is_available
from torch.cuda import is_bf16_supported
from torch import bfloat16, float16, dtype


# dataset
@dataclass
class DATASET:
    HG_TOKEN:str = ""
    DATASET:str = "/nonsense_beta/datasets/your_dataset"
    PATH_CACHE:Optional[str] = None
    TEST:bool = True

class DATASETS(object):
    def __init__(self, datasets:Optional[DATASET|List[DATASET]]=None):
        if datasets:
            if isinstance(datasets, DATASET):
                self._datasets = [datasets, ]
            elif isinstance(datasets, list):
                self._datasets = datasets
            else:
                raise Exception(f"argument error: need DATASET or [DATASET, ...], got {type(datasets)}")
        else:
            self._datasets = [DATASET(), ]
    
    def __call__(self, datasets:Optional[DATASET|List[DATASET]]=None):
        if datasets:
            if isinstance(datasets, DATASET):
                self._datasets += [datasets, ]
            elif isinstance(datasets, list):
                self._datasets += datasets
            else:
                raise Exception(f"argument error: need DATASET or [DATASET, ...], got {type(datasets)}")
        else:
            self._datasets += [DATASET(), ]
    
    def __iter__(self):
        self._iterator = iter(self._datasets)
        return self
    
    def __next__(self):
        return next(self._iterator)

#tokenizer
@dataclass
class TOKENIZER:
    _INPUT:str = ""
    PATH_TOKENIZER_MODEL:str = "/nonsense/datasets/base_on_your_dataset_tokenizer_model/tokenizer_model.model"
    VOCABSIZE:int = 2**17  # 131,072
    _CHARACTER_COVERAGE:float = 0.9995
    _MODEL_TYPE:str = "bpe"
    CMD_TRAIN:str = f"spm_train --input={_INPUT} --model_prefix={PATH_TOKENIZER_MODEL} --vocab_size={VOCABSIZE} --character_coverage={_CHARACTER_COVERAGE} --model_type={_MODEL_TYPE}"

# model
@dataclass
class MODEL:
    '''
    emb_dim==model_dim
        _ = input:[batch, seq_len]
    --> _ = embedding(_):[batch, seq_len, emb_dim]
    --> _ = _*sqrt(emb_dim):[batch, seq_len, emb_dim]
    --> pos:[seq_len, 1]
        multiplication = 1/(10,000^(2i/d_model))
                       = exp(-2i*(log(10,000)/d_model))
        pos_enc = even(sin(pos*multiplication))|odd(cos(pos*multiplication)):[seq_len, emb_dim]
        _ = _+pos_enc:[batch, seq_len, emb_dim]
    --> transformer:[emb_dim, emb_dim]
        _ = transformer(_):[batch, seq_len, emb_dim]
    '''
    SEQ_LEN:int = 10
    MODEL_DIM:int = 512
    EMBEDDING_NUM:int = TOKENIZER.VOCABSIZE
    EMBEDDING_DIM:int = MODEL_DIM
    HEAD_NUM:int = 8
    ENCODERLAYERS_NUM:int = 6
    DECODERLAYERS_NUM:int = 6
    FEEDFORWARD_DIM:int = 2048
    DROPOUTPROB:float = 0.1

    def asdict(self) -> Dict[str, int|float]:
        _ = dict(seqLen=self.SEQ_LEN,
                 embNum=self.EMBEDDING_NUM,
                 embDim=self.EMBEDDING_DIM,
                 headNum=self.HEAD_NUM, 
                 encoderLayersNum=self.ENCODERLAYERS_NUM, 
                 decoderLayersNum=self.DECODERLAYERS_NUM, feedforwardDim=self.FEEDFORWARD_DIM, 
                 dropoutProb=self.DROPOUTPROB)
        return _

@dataclass
class TRAIN:
    COMPILE:bool = False
    DEVICE:str = accelerator_is_available() and current_accelerator().type or "cpu"
    TYPE:dtype = cuda_is_available and is_bf16_supported and bfloat16 or float16
    BATCH:int = 12
    LEARNING_RATE:float = 1e-3
    EPOCHS:int = 1
    SAVE_PATH:str = "/nonsense_beta/checkpoints"