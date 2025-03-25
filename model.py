from torch import nn
import math
import torch


class StreamLayer(nn.Module):
    def __init__(self, seqLen:int, embNum:int, embDim:int, dropoutProb:float):
        super().__init__()
        self._seqLen = seqLen
        self._embDim = embDim
        self._modelDim = embDim
        self._embeddingLayer = nn.Embedding(num_embeddings=embNum, embedding_dim=embDim)
        self._dropoutLayer = nn.Dropout(p=dropoutProb)

    def _positionalEncoding(self):
        position = torch.arange(start=0, end=self._seqLen, step=1).view(-1, 1)
        multiplication = torch.exp(-torch.arange(start=0, end=self._embDim*2, step=2, dtype=torch.float32)*math.log(10000)/self._modelDim)
        excessive = position*multiplication
        _ = torch.zeros(self._seqLen, self._embDim, device=self._embeddingLayer.weight.device)
        _[:, 0::2] = torch.sin(excessive[:, 0::2])
        _[:, 1::2] = torch.cos(excessive[:, 1::2])
        return _.to(device=self._embeddingLayer.weight.device, dtype=self._embeddingLayer.weight.dtype)

    def forward(self, stream):
        return self._dropoutLayer(self._embeddingLayer(stream.to(self._embeddingLayer.weight.device))*math.sqrt(self._modelDim)+self._positionalEncoding())

class NonsenseModel(nn.Transformer):
    def __init__(self, 
                 seqLen:int,
                 embNum:int,
                 embDim:int,
                 headNum:int, 
                 encoderLayersNum:int, decoderLayersNum:int, feedforwardDim:int, 
                 dropoutProb:float):
        super().__init__(d_model=embDim,
                         nhead=headNum,
                         num_encoder_layers=encoderLayersNum,
                         num_decoder_layers=decoderLayersNum,
                         dim_feedforward=feedforwardDim,
                         dropout=dropoutProb,
                         activation=nn.functional.gelu,
                         batch_first=True,
                         norm_first=True,
                         bias=True)
        self._seqLen = seqLen
        self._modelDim = embDim
        self._inputLayer = StreamLayer(seqLen=seqLen, 
                                       embNum=embNum, 
                                       embDim=embDim, 
                                       dropoutProb=dropoutProb)
        self._outputLayer = StreamLayer(seqLen=seqLen, 
                                       embNum=embNum, 
                                       embDim=embDim, 
                                       dropoutProb=dropoutProb)
        self._linear = nn.Linear(embDim, embNum)
    
    def forward(self, src, tgt):
        _ = super().forward(src=self._inputLayer(src),
                             tgt=self._outputLayer(tgt), 
                             memory_mask=self.generate_square_subsequent_mask(self._seqLen), 
                             memory_is_causal=True)
        return nn.functional.log_softmax(self._linear(_), dim=-1)