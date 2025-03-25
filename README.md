# nonsense

## Introduction
This project is a toy. It's roughly implemented following the Attention is All You Need paper. 

## ISSUES
1. We can found the loss is very low at the beginning of the training. I guess it's over fit, because dataset.py::Datasets.getData use torch.utils.data.DataLoader sampling by a serial continuous int. It causes a lot of duplicate data.
2. I didn't test the training code carefully.
3. Many code should be optimized.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
