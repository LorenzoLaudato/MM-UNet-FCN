# MM-U-Net: Semantic segmentation and pattern classification using a multi-channel mask 

- [Quick start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Data](#data)

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
**Note : Use Python 3.6 or newer**

### Training

```console
> python3 train_experiments.py -h

usage: train_experiments.py [-h] [--epochs E] [--batch-size B]
                            [--learning-rate LR] [--load LOAD]
                            [--load_data LOAD_DATA] [--save_data SAVE_DATA]
                            [--amp] [--bilinear] [--classes CLASSES]
                            [--channels CHANNELS]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --load_data LOAD_DATA
                        Load dataloaders from a .pth file
  --save_data SAVE_DATA
                        Save dataloaders into a .pth file
  --amp                 Use mixed precision
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
  --channels CHANNELS, -n CHANNELS
                        Number of input img channels

```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

```console
> python3 test_experiments.py -h
usage: test_experiments.py [-h] [--batch-size B] [--load LOAD]
                           [--load_data LOAD_DATA] [--save_data SAVE_DATA]
                           [--bilinear] [--classes CLASSES]
                           [--channels CHANNELS] [--fold FOLD]
                           [--visualize VISUALIZE] [--spec_level SPEC_LEVEL]
                           [--mathematical_morph MATHEMATICAL_MORPH]

Test the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --batch-size B, -b B  Batch size
  --load LOAD, -m LOAD  Load model from a .pth file
  --load_data LOAD_DATA
                        Load dataloaders from a .pth file
  --save_data SAVE_DATA
                        Save dataloaders into a .pth file
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
  --channels CHANNELS, -n CHANNELS
                        Number of input img channels
  --fold FOLD           Number of experiment
  --visualize VISUALIZE, -v VISUALIZE
                        Visualize the outputs
  --spec_level SPEC_LEVEL, -sl SPEC_LEVEL
                        Test on specimen level
  --mathematical_morph MATHEMATICAL_MORPH, -mm MATHEMATICAL_MORPH
                        Apply mathematical morphological operators on the
                        output



You can specify which model file to use with `--model MODEL.pth`.


Available scales are 0.5 and 1.0.

## Data
You can use your own dataset as long as you make sure it is loaded properly in `utils/dataloader.py` or `utils/dataloader_online.py` .

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
