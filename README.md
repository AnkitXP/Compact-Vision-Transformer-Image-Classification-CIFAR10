# Image Classification of CIFAR-10 using Compact Convolutional Transformers

### Overview
This project demonstrates the use of Compact Convolutional Transformers (CCT) for image classification on the CIFAR-10 dataset. Unlike traditional CNNs or large-scale transformers, Compact Transformers offer a highly efficient model with fewer parameters, making them ideal for small-scale datasets and researchers with limited computational resources.

### Project Details
- Dataset: CIFAR-10
- Model: Compact Convolutional Transformer (CCT)
- Purpose: To improve the Vision Transformer’s performance on low-resolution datasets by combining CNN and transformer layers.

### Methodology
The implementation uses CNN layers as a preprocessing embedding to feed into transformer layers, which helps in feature extraction from convoluted and pooled images. Max pooling is applied to obtain essential features, followed by a series of transformer layers.

### Key techniques include:

CNN Convolutions as an embedding layer.
Max Pooling for feature extraction.
Random Data Augmentation, CutMix, and MixUp for improved generalization.

### Hyperparameters
- Image Size: 32px
- Input Channels: 3
- Kernel Size: 2
- Depth of Attention: 7
- Number of Heads: 4
- Dropouts: 0.1 across layers
- Hidden Dimensions: 64
- Batch Size: 64
- Learning Rate: 1e-3
- Optimizer: AdamW with a custom learning rate scheduler (warmup and cosine decay)

### Training
The model is trained over 200 epochs. Training metrics, including accuracy and loss, are tracked across epochs. The use of extensive data augmentation has led to better validation scores over training scores, showing the model's generalization capabilities.

### Prerequisites:
- Python 3.6+
- PyTorch 1.0+

```conda env create evironment.yml```

### Command to run on train mode:
```cd code/```
```python main.py --mode train```

### Command to run on test mode:
```python main.py --mode test --load <checkpoint name>```


### Command to run on predict mode:
```python main.py --mode predict checkpoint name>```

Kindly replace the checkpoint file names wherever required.

### Results
The model achieves a validation accuracy of 87%, showing promising results for a compact model on a low-resolution dataset. With additional epochs and optimization, there is room for further improvement.

### Conclusion
Compact Transformers have proven to be effective for small datasets like CIFAR-10, achieving competitive accuracy with a small model size. Future work could focus on tuning the learning rate schedule or experimenting with advanced optimizers to further improve performance.

### References
- Escaping the Big Data Paradigm with Compact Transformers (https://arxiv.org/pdf/2104.05704)
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (https://arxiv.org/pdf/2010.11929)
