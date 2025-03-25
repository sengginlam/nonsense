from os import path
from subprocess import call
from PARAMETER import TOKENIZER
from typing import List
import sentencepiece as spm


class NoTokenizerModelFound(Exception):
    pass

class Tokenizer(object):
    def __init__(self, config:TOKENIZER):
        if not path.exists(config.PATH_TOKENIZER_MODEL):
            print("training tokenizer model...")
            if call(config.CMD_TRAIN, shell=True)==0:
                print(f"DONE!\ntokenizer model saved as {config.PATH_TOKENIZER_MODEL}")
            else:
                print("ERROR EXIT: something WRONG")
                raise NoTokenizerModelFound
        self._sp = spm.SentencePieceProcessor(model_file=config.PATH_TOKENIZER_MODEL)
    
    def encode(self, text:str|List[str]) -> List[int]:
        return self._sp.Encode(text, out_type=int)

    def decode(self, tokens:List[int]) -> str:
        return self._sp.Decode(tokens, out_type=str)