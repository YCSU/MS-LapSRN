from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(3)

from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Add, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


filters = 32
D = 5
R = 3

def embedding_model(filters=filters, layer_num=D, name="embed"):
    '''
    Extracting features inside the shared module
    '''
    img_input = Input(shape=(None, None, filters))
    x = LeakyReLU(alpha=0.2)(img_input)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    for _ in range(layer_num-1):
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, (3,3), padding='same',
                    use_bias=False)(x)
    model = Model(inputs=img_input, outputs=x, name=name)
    return model


def _upsample_and_condense(x, filters=filters):
    '''
    upsample and condense to resudual image 
    inside the shared module
    '''
    x = LeakyReLU(alpha=0.2)(x)
    upsample = Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same',
                        use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(upsample)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    condense = Conv2D(1, (3,3), padding='same',
                        use_bias=False)(x)
    return condense, upsample

def residual_model(recursive_num=R, filters=filters):
    '''
    pipeline for the shared module
    '''
    embedding = embedding_model()
    img_input = Input(shape=(None, None, filters))
    x = embedding(img_input)
    x = Add()([x, img_input])
    for _ in range(recursive_num-1):
        x = embedding(x)
        x = Add()([x, img_input])
    res, upsample = _upsample_and_condense(x)
    model = Model(inputs=img_input, outputs=[res, upsample], name="residual")
    return model

def upsample_model(filters=filters):
    '''
    upsmaple the input image
    '''
    img_input = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    upsample = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same',
                        use_bias=False)(x)
    model = Model(inputs=img_input, outputs=upsample, name="upsample")
    return model

def initConv_model(filters=filters):
    '''
    transform the original image into a set of feature maps
    '''
    img_input = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    model = Model(inputs=img_input, outputs=output, name="init_embedd")
    return model

def net():
    initConv = initConv_model()
    residual = residual_model()
    upsample = upsample_model()

    img_input = Input(shape=(None, None, 1))
    
    # x2
    embedded_x = initConv(img_input)
    upsample_1 = upsample(img_input)
    residual_1, f_upsample_1 = residual(embedded_x)
    hr1 = Add()([upsample_1, residual_1])

    # x4
    upsample_2 = upsample(hr1)
    residual_2, f_upsample_2 = residual(f_upsample_1)
    hr2 = Add()([upsample_2, residual_2])
    model = Model(inputs=img_input, outputs=[hr1, hr2])
    return model

