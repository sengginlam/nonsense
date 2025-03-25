# nonsense

## Introduction
This project is a toy. It's roughly implemented following the Attention is All You Need paper. 

## ISSUES
1. We can find the loss is very low at the beginning of the training. I guess it's overfitting because dataset.py::Datasets.getData uses torch.utils.data.DataLoader sampling by a serial continuous int. It causes a lot of duplicate data.
2. I didn't test the training code carefully.
3. Many codes should be optimized.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
