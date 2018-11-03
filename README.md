# MS-LapSRN
Multi-Scale Laplacian Super Resolution Network

This is an Keras implementation based on the following research with a few tweaks,  
[Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks](http://vllab.ucmerced.edu/wlai24/LapSRN/).
The model has multi-scale training, shared-source skip connection and shared parametes for each upscale stage. With these  properties, it achieves a depth of ~40 layers with just ~120,000 trainable parameters (32 filters for each 3x3 convolutional layer). Because of this, it is suitable for small datasets.

## How to train the model

### model
The model is defined in *msLapSRN_model.py*. Current implementation increases the resolution by 16 times (4x for width and height).

### data
The data is read in with a [Sequence class](https://keras.io/utils/) for training with the **fit_generator** function. The template is in *dataset.py*. The data should be stored in the [HDF5](https://www.h5py.org/) format with the following structure

* *train/data*       --- low resolution images      
* *train/label_x2*   --- 2x images
* *train/label_x4*   --- 4x images

### training
Finally, the L1 Charbonnier loss and the training pipeline is defined in *train.py*. Once the training data is structured as described above, the model can be trained by just
```
python train.py
```




