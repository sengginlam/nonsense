from dataset import Datasets
from torch import compile, no_grad, save
from torch import float as tfloat
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from PARAMETER import DATASETS, TRAIN, MODEL
from model import NonsenseModel


trainConfig = TRAIN()
modelConfig = MODEL()
modelConfigDict = modelConfig.asdict()

model = NonsenseModel(**modelConfigDict).to(trainConfig.DEVICE, trainConfig.TYPE)
model = compile(model) if trainConfig.COMPILE else model
lossFn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=trainConfig.LEARNING_RATE)

datasets = iter(Datasets(DATASETS()))

for epoch in range(trainConfig.EPOCHS):
    while next(datasets):
        # train
        datasets.train()
        datasetSize = len(datasets)
        model.train()
        for batch, (x, y) in enumerate(datasets.getData()):
            pred = model(x, y)
            loss = lossFn(pred, one_hot(y, modelConfig.EMBEDDING_NUM).to(device=trainConfig.DEVICE, dtype=trainConfig.TYPE))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch%1000==0:
                loss = loss.item()
                current = (batch+1)*len(x)
                print(f"LOSS: {loss:>16f}\tCURRENT: {current:>8d} / {datasetSize:>8d}")
        
        # val
        datasets.val()
        datasetSize = len(datasets)
        model.eval()
        testLoss = 0
        correct = 0
        with no_grad():
            for batch, (x, y) in enumerate(datasets.getData()):
                pred = model(x, y)
                testLoss += lossFn(pred, one_hot(y, modelConfig.EMBEDDING_NUM).to(device=trainConfig.DEVICE, dtype=trainConfig.TYPE)).item()
                correct += (pred.argmax(0)==y).type(tfloat).sum().item()
        testLoss /= trainConfig.BATCH_SIZE
        correct /= datasetSize
        print(f"ACCURACY: {correct*100:>0.2f}%\tAVG_LOSS: {testLoss:>8f}")

save(model.state_dict(), trainConfig.SAVE_PATH)