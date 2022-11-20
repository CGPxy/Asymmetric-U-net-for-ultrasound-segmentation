# coding=utf-8
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *                                         
from model_function import *


def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data) #,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2

def ConvBlock1(data, data1,filte):
    concatenate = Concatenate()([data, data1])
    conv1 = Conv2D(filte, (3, 3), padding="same")(concatenate) #,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2


def updata1(filte, data, skipdata):
    g1 = UnetGatingSignal(data, is_batchnorm=True)
    attn1 = AttnGatingBlock(skipdata, g1, 128)
    up1 = concatenate([Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(data), attn1])

    up1 = Conv2D(filte, (3, 3), padding="same")(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.01)(up1)
    up1 = squeeze_excitation_layer(data=up1, ratio=4, out_dim=filte)

    conv1 = Conv2D(filte, (3, 3), padding="same")(up1)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2

def updata(filte, data, skipdata, skipdata1):
    data1 = UpSampling2D((2, 2))(data)
    skipdata11 = UpSampling2D((2, 2))(skipdata1)
    skipdata0 = Concatenate()([skipdata,skipdata11])

    g1 = UnetGatingSignal(data1, is_batchnorm=True)
    attn1 = AttnGatingBlock(skipdata0, g1, 128)
    up1 = concatenate([Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(data1), attn1])

    up1 = Conv2D(filte, (3, 3), padding="same")(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.01)(up1)
    up1 = squeeze_excitation_layer(data=up1, ratio=4, out_dim=filte)

    conv1 = Conv2D(filte, (3, 3), padding="same")(up1)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2


def soout(data, size,name):
    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(data)
    out = Activation('sigmoid')(outconv)
    up = UpSampling2D(size=size,name=name)(out)

    return up


def Network():
    inputs = Input((img_h, img_w, 3))

    Conv1 = ConvBlock(data=inputs, filte=64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = ConvBlock(data=pool1, filte=128)

    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    pool21 = MaxPooling2D(pool_size=(4, 4))(Conv1)
    pool21 = Conv2D(32, (3, 3), padding="same")(pool21)
    pool21 = BatchNormalization()(pool21)
    pool21 = LeakyReLU(alpha=0.01)(pool21)
    Conv3 = ConvBlock1(data=pool2, data1=pool21, filte=128)

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)    
    Conv4 = ConvBlock(data=pool3, filte=256)

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)
    pool41 = MaxPooling2D(pool_size=(4, 4))(Conv3) 
    pool41 = Conv2D(32, (3, 3), padding="same")(pool41)
    pool41 = BatchNormalization()(pool41)
    pool41 = LeakyReLU(alpha=0.01)(pool41)  
    Conv5 = ConvBlock1(data=pool4, data1=pool41,filte=256)

    pool5 = MaxPooling2D(pool_size=(2, 2))(Conv5)    
    Conv6 = ConvBlock(data=pool5, filte=512)

    pool6 = MaxPooling2D(pool_size=(2, 2))(Conv6) 
    pool61 = MaxPooling2D(pool_size=(4, 4))(Conv5) 
    pool61 = Conv2D(32, (3, 3), padding="same")(pool61)
    pool61 = BatchNormalization()(pool61)
    pool61 = LeakyReLU(alpha=0.01)(pool61)  
    Conv7 = ConvBlock1(data=pool6, data1=pool61, filte=512)

    pool7 = MaxPooling2D(pool_size=(2, 2))(Conv7)    
    Conv8 = ConvBlock(data=pool7, filte=1024)

    # 6
    up1 = updata1(filte=512, data=Conv8, skipdata=Conv7)
    out4 = soout(data=up1, size=(64,64),name='out4')
    
    # 24
    up2 = updata(filte=256, data=up1, skipdata=Conv5, skipdata1=Conv6)
    out3 = soout(data=up2, size=(16,16),name='out3')

    # 96
    up3 = updata(filte=128, data=up2, skipdata=Conv3, skipdata1=Conv4)
    out2 = soout(data=up3, size=(4,4),name='out2')

    # 384
    up4 = updata0(filte=64, data=up3, skipdata=Conv1, skipdata1=Conv2)
    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up4)
    out = Activation('sigmoid',name='out')(outconv)

    model = Model(inputs=inputs, outputs=[out,out2,out3,out4])
    return model