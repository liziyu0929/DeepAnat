# cnn_models
#
# (c) Ziyu Li, Qiyuan Tian, Artificial Intelligence in Neuroimaging Software, 2022

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, concatenate, Multiply, BatchNormalization, MaxPooling3D, UpSampling3D, Activation,\
                        Dense, Conv2D, LeakyReLU, Multiply, Flatten, Dropout, add, Permute, concatenate, Lambda
from tensorflow.keras.initializers import RandomNormal

num_slice = 64

def conv3d_bn_relu(inputs, filter_num, bn_flag=False):
    if bn_flag:
        conv = Conv3D(filter_num, (3,3,3), padding='same', kernel_initializer='he_normal')(inputs)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)        
    else:
        conv = Conv3D(filter_num, (3,3,3), padding='same', kernel_initializer='he_normal')(inputs)
        conv = Activation('relu')(conv)
    return conv

def unet_3d_model(num_ch=1, filter_num=48, kinit_type='he_normal', tag='unet3d'):
    
    inputs = Input((None, None, None, num_ch)) 
    loss_weights = Input((None, None, None, 1))
    
    p0 = inputs
    
    conv1 = conv3d_bn_relu(p0, filter_num)
    conv1 = conv3d_bn_relu(conv1, filter_num)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = conv3d_bn_relu(pool1, filter_num*2)
    conv2 = conv3d_bn_relu(conv2, filter_num*2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = conv3d_bn_relu(pool2, filter_num*4)
    conv3 = conv3d_bn_relu(conv3, filter_num*4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = conv3d_bn_relu(pool3, filter_num*8)
    conv4 = conv3d_bn_relu(conv4, filter_num*8)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3d_bn_relu(pool4, filter_num*16)
    conv5 = conv3d_bn_relu(conv5, filter_num*16)

    up6 = UpSampling3D(size = (2, 2, 2))(conv5)
    merge6 = concatenate([conv4,up6])
    conv6 = conv3d_bn_relu(merge6, filter_num*8)
    conv6 = conv3d_bn_relu(conv6, filter_num*8)
    
    up7 = UpSampling3D(size = (2, 2, 2))(conv6)
    merge7 = concatenate([conv3,up7])
    conv7 = conv3d_bn_relu(merge7, filter_num*4)
    conv7 = conv3d_bn_relu(conv7, filter_num*4)

    up8 = UpSampling3D(size = (2, 2, 2))(conv7)
    merge8 = concatenate([conv2,up8])
    conv8 = conv3d_bn_relu(merge8, filter_num*2)
    conv8 = conv3d_bn_relu(conv8, filter_num*2)

    up9 = UpSampling3D(size = (2, 2, 2))(conv8)
    merge9 = concatenate([conv1,up9])
    conv9 = conv3d_bn_relu(merge9, filter_num)
    conv9 = conv3d_bn_relu(conv9, filter_num)
    
    recon = Conv3D(1, (3, 3, 3), padding='same',
                  activation=None, 
                  kernel_initializer='he_normal'
                  )(conv9)
            
    recon = concatenate([recon, loss_weights],axis=-1)
    
    model = Model(inputs=[inputs, loss_weights], outputs=recon) 

    return model

def discriminator_2d_model(img_size, num_ch=1, tag='discriminator2d'):
    
    inputs = Input((img_size, img_size, num_ch)) # channel last
    initializer = RandomNormal(mean=0., stddev=0.02)
    
    df_dim = 64
    
    net_in = inputs
    net_h0 = Conv2D(df_dim, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_in)
    net_h0 = LeakyReLU(alpha=0.2)(net_h0)
    net_h0 = Conv2D(df_dim, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h0)
    net_h0 = BatchNormalization()(net_h0)
    net_h0 = LeakyReLU(alpha=0.2)(net_h0)

    net_h1 = Conv2D(df_dim * 2, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h0)
    net_h1 = BatchNormalization()(net_h1)
    net_h1 = LeakyReLU(alpha=0.2)(net_h1)
    net_h1 = Conv2D(df_dim * 2, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h1)
    net_h1 = BatchNormalization()(net_h1)
    net_h1 = LeakyReLU(alpha=0.2)(net_h1)
    
    net_h2 = Conv2D(df_dim * 4, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h1)
    net_h2 = BatchNormalization()(net_h2)
    net_h2 = LeakyReLU(alpha=0.2)(net_h2)
    net_h2 = Conv2D(df_dim * 4, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h2)
    net_h2 = BatchNormalization()(net_h2)
    net_h2 = LeakyReLU(alpha=0.2)(net_h2)
    
    net_h3 = Conv2D(df_dim * 8, (3, 3), strides=(1, 1), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h2)
    net_h3 = BatchNormalization()(net_h3)
    net_h3 = LeakyReLU(alpha=0.2)(net_h3)
    net_h3 = Conv2D(df_dim * 8, (3, 3), strides=(2, 2), activation=None, use_bias=False, padding='same',kernel_initializer=initializer)(net_h3)
    net_h3 = BatchNormalization()(net_h3)
    net_h3 = LeakyReLU(alpha=0.2)(net_h3)
    
    net_ho = Flatten()(net_h3)
    net_ho = Dense(df_dim * 16, activation=None)(net_ho)
    net_ho = LeakyReLU(alpha=0.2)(net_ho)
    net_out = Dense(1, activation='sigmoid')(net_ho)
    
    model = Model(inputs = inputs, outputs = net_out)
    
    return model

def slice_2d(x,index):
    return x[:,:,:,:index]

def slice(x,index):
    return x[:,:,:,:,:index]

def squeeze_first2axes_operator(x5d) :
    shape = tf.shape(x5d) # get dynamic tensor shape
    x4d = tf.reshape(x5d, [shape[0]*shape[1], shape[2], shape[3], shape[4]])
    return x4d

def squeeze_first2axes_shape(x5d_shape):
    in_batch, in_slice, in_rows, in_cols, in_filters = x5d_shape
    if (in_batch is None):
        output_shape = (None, in_rows, in_cols, in_filters)
    else:
        output_shape = (in_batch*in_slice, in_rows, in_cols, in_filters)
    return output_shape

def flatten_first2axes_operator_1d(x2d):
    shape = tf.shape(x2d) # get dynamic tensor shape
    x3d = tf.reshape(x2d, [shape[0]//(num_slice), num_slice, shape[1]])
    return x3d

def flatten_first2axes_operator_3d(x2d):
    shape = tf.shape(x2d) # get dynamic tensor shape
    x3d = tf.reshape(x2d, [shape[0]//(num_slice*3), num_slice*3, shape[1]])
    return x3d

def flatten_first2axes_shape_1d(x2d_shape):
    in_batch, in_filters = x2d_shape
    if (in_batch is None):
        output_shape = (None, num_slice, in_filters)
    else:
        output_shape = (in_batch//(num_slice), num_slice, in_filters)
    return output_shape

def flatten_first2axes_shape_3d(x2d_shape):
    in_batch, in_filters = x2d_shape
    if (in_batch is None):
        output_shape = (None, num_slice * 3, in_filters)
    else:
        output_shape = (in_batch//(num_slice), num_slice * 3, in_filters)
    return output_shape

def hybrid_gan_model(block_size, input_ch_g, input_ch_d, generator, discriminator,tag='hybrid_gan'):
    inputs = Input((block_size[0], block_size[1], block_size[2], input_ch_g))  
    bmask = Input((block_size[0], block_size[1], block_size[2], 1))
    loss_weights = Input((block_size[0], block_size[1], block_size[2], 1))
    
    generated_block = generator([inputs, bmask, loss_weights])
    
    generated_block_1 = Lambda(slice, arguments={'index':input_ch_d})(generated_block)
    generated_block_2 = Permute(dims=(2,1,3,4))(generated_block_1)
    generated_block_3 = Permute(dims=(3,1,2,4))(generated_block_1)
    generated_blocks = concatenate([generated_block_1, generated_block_2, generated_block_3],axis=0)
    generated_images = Lambda(squeeze_first2axes_operator, output_shape = squeeze_first2axes_shape)(generated_blocks)
    discriminator_outputs = discriminator(generated_images)
    discriminator_outputs = Lambda(flatten_first2axes_operator_3d, 
                                   output_shape = flatten_first2axes_shape_3d)(discriminator_outputs)
    
    model = Model(inputs=[inputs, bmask, loss_weights], outputs=[generated_block, discriminator_outputs])
    return model