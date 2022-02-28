# s_DeepAnat_trainGAN.py
#
# (c) Ziyu Li, Qiyuan Tian, Artificial Intelligence in Neuroimaging Software, 2022

# %% load moduals
import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import tensorflow as tf

from keras.optimizers import Adam

# for compatibility
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import aini_utils as utils
from cnn_models import unet_3d_model, discriminator_2d_model, hybrid_gan_model

# %% set up path
dpRoot = os.path.dirname(os.path.abspath('s_DeepAnat_trainGAN.py'))
os.chdir(dpRoot)

# %% subjects
dpData = '/autofs/space/rhapsody_001/users/qtian/AINI/DeepAnat'
subjects = sorted(glob.glob(os.path.join(dpData, 'Tract*')))

# %% load data 
train_block_in = np.array([])
valid_block_in = np.array([])

sz_block = 64    
sz_pad = 1
flip = 1 # flip along x to augment training data
input_list = ['diff_meanb0', 'diff_meandwi', 'diff_dtiL1', 'diff_dtiL2', 'diff_dtiL3',
              'diff_dtiDwi1', 'diff_dtiDwi2', 'diff_dtiDwi3', 'diff_dtiDwi4', 'diff_dtiDwi5', 'diff_dtiDwi6']

for ii in np.arange(len(subjects)):
    sj = os.path.basename(subjects[ii])
    
    print(sj)
    dpSub = os.path.join(dpData, sj)
    
    fpT1w = os.path.join(dpSub, sj + '_t1w.nii.gz')
    t1w = nb.load(fpT1w).get_data()   
    t1w = np.expand_dims(t1w, -1)

    fpMask = os.path.join(dpSub, sj + '_mask.nii.gz')
    mask = nb.load(fpMask).get_data() 
    mask = np.expand_dims(mask, -1)
        
    input = 0.
    for jj in np.arange(0, len(input_list)):
        
        fpImage = os.path.join(dpSub, sj + '_' + input_list[jj] + '.nii.gz')
        image = nb.load(fpImage).get_data()   
        image = np.expand_dims(image, -1)      
        
        if jj == 0:
            inputs = image
        else:
            inputs = np.concatenate((inputs, image), axis=-1)

    norm_ch = [0, 1, 5, 6, 7, 8, 9, 10]
    t1w_norm, tmp = utils.normalize_image(t1w, t1w, mask)
    inputs_norm, tmp = utils.normalize_image(inputs, inputs, mask, norm_ch)
    
    t1w_norm = t1w_norm * mask # exclude non-brain content from loss calculation
    inputs_norm = inputs_norm * mask
    
    ind_block, ind_brain = utils.block_ind(mask, sz_block=sz_block, sz_pad=sz_pad)
    
    t1w_norm_block = utils.extract_block(t1w_norm, ind_block)
    inputs_norm_block = utils.extract_block(inputs_norm, ind_block)
    mask_block = utils.extract_block(mask, ind_block)
    
    t1w_norm_block = np.concatenate((t1w_norm_block, mask_block), axis=-1)
    
    if flip: # Flip x to augment data
        inputs_norm_block_flip = inputs_norm_block[:,::-1,:,:,:]
        mask_block_flip = mask_block[:,::-1,:,:,:]
        t1w_norm_block_flip = t1w_norm_block[:,::-1,:,:,:]
        
        inputs_norm_block = np.concatenate((inputs_norm_block, inputs_norm_block_flip), axis=0)
        mask_block = np.concatenate((mask_block, mask_block_flip), axis=0)
        t1w_norm_block = np.concatenate((t1w_norm_block, t1w_norm_block_flip), axis=0)
    
    if np.mod(ii + 2, 5) == 0: # 1 out of 5 subjects for validation
        print('validation subject')
        
        if valid_block_in.size == 0: 
            valid_block_out = t1w_norm_block
            valid_block_in = inputs_norm_block      
            valid_block_mask = mask_block     
        else:
            valid_block_out = np.concatenate((valid_block_out, t1w_norm_block), axis=0)
            valid_block_in = np.concatenate((valid_block_in, inputs_norm_block), axis=0)
            valid_block_mask = np.concatenate((valid_block_mask, mask_block), axis=0)
    else:
        print('training subject')

        if train_block_in.size == 0: 
            train_block_out = t1w_norm_block
            train_block_in = inputs_norm_block           
            train_block_mask = mask_block     
        else:
            train_block_out = np.concatenate((train_block_out, t1w_norm_block), axis=0)
            train_block_in = np.concatenate((train_block_in, inputs_norm_block), axis=0)
            train_block_mask = np.concatenate((train_block_mask, mask_block), axis=0)

# %% view data
plt.imshow(train_block_out[47, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_out[47, :, :, 40, 1], clim=(0., 1), cmap='gray')
plt.imshow(train_block_mask[47, :, :, 40, 0], clim=(0, 1), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 2], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 3], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 4], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 5], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 6], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 7], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 8], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 9], clim=(-2., 2.), cmap='gray')
plt.imshow(train_block_in[47, :, :, 40, 10], clim=(-2., 2.), cmap='gray')

# set up models
input_ch_g = train_block_in.shape[-1]
input_ch_d = 1

model_generator = unet_3d_model(input_ch_g)
model_generator.summary()
model_discriminator = discriminator_2d_model(sz_block, input_ch_d)
model_discriminator.summary()
model_discriminator.trainable = False
model_gan = hybrid_gan_model(sz_block, input_ch_g, input_ch_d, model_generator, model_discriminator)
model_gan.summary()

# set up optimizer
opt_g = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
opt_d = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

# compile models
model_generator.compile(loss = utils.mean_absolute_error_weighted, optimizer = opt_g)
model_discriminator.trainable = True
model_discriminator.compile(loss = 'binary_crossentropy', optimizer = opt_d)
model_discriminator.trainable = False
loss = [utils.mean_absolute_error_weighted, 'binary_crossentropy']
loss_weights = [1, 1e-3]
model_gan.compile(optimizer = opt_g, loss = loss, loss_weights=loss_weights)
model_discriminator.trainable = True
    
# train
num_epochs = 50
label_smoothing_factor = 1
vgg_loss =  []
l1_loss_train, l1_loss_test = [], []
gan_loss_train, gan_loss_test = [], []
d_loss_train, d_loss_test = [], []

fnCp = 'gan_all2t1'

total_train_num = train_block_out.shape[0]
total_test_num = valid_block_out.shape[0]
print('Training on', total_train_num, 'blocks. Testing on', total_test_num, 'blocks.')
batch_size_train, batch_size_test = 1, 1

for ii in range(num_epochs):
    cnt_train, cnt_test = 0, 0
    
    # shuffle the images
    index_train = np.arange(total_train_num)
    np.random.shuffle(index_train)
    train_block_in = train_block_in[index_train,:,:,:,:]
    train_block_mask = train_block_mask[index_train,:,:,:,:]
    train_block_out = train_block_out[index_train,:,:,:,:]

    index_test = np.arange(total_test_num)
    np.random.shuffle(index_test)
    valid_block_in = valid_block_in[index_test,:,:,:,:]
    valid_block_mask = valid_block_mask[index_test,:,:,:,:]
    valid_block_out = valid_block_out[index_test,:,:,:,:]

    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('\n')
    print('Epoch count:', ii + 1)
    gan_loss_train_batch, l1_loss_train_batch, d_loss_train_batch = [], [], []
    gan_loss_test_batch, l1_loss_test_batch, d_loss_test_batch = [], [], []
    while cnt_train + batch_size_train < total_train_num:
        if cnt_test + batch_size_test >= total_test_num:
            cnt_test = 0
        
        print('\n')
        print('Training blocks count:', cnt_train)
        
        # prepare training and testing batch
        train_batch_input = train_block_in[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        train_batch_output = train_block_out[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        train_batch_bmask = train_block_mask[cnt_train:cnt_train+batch_size_train,:,:,:,:]
        
        test_batch_input = valid_block_in[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        test_batch_output = valid_block_out[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        test_batch_bmask = valid_block_mask[cnt_test:cnt_test+batch_size_test,:,:,:,:]
        
        # prepare labels and images for discriminator
        if ii == 0 and cnt_train == 0:
            t1w_generated_train = train_batch_input[:,:,:,:,:input_ch_d]
            t1w_generated_test = test_batch_input[:,:,:,:,:input_ch_d]
    
        else:
            t1w_generated_train = model_generator.predict([train_batch_input, train_batch_bmask])[:,:,:,:,:input_ch_d]
            t1w_generated_test = model_generator.predict([test_batch_input, test_batch_bmask])[:,:,:,:,:input_ch_d]
        
        t1w_generated_train_sag = t1w_generated_train
        t1w_generated_train_cor = np.transpose(t1w_generated_train,[0,2,1,3,4]) # generate images of different directions
        t1w_generated_train_axial = np.transpose(t1w_generated_train,[0,3,1,2,4])
        t1w_generated_test_sag = t1w_generated_test
        t1w_generated_test_cor = np.transpose(t1w_generated_test,[0,2,1,3,4])
        t1w_generated_test_axial = np.transpose(t1w_generated_test,[0,3,1,2,4])
        t1w_generated_train_all = np.concatenate((t1w_generated_train_sag, 
                                                t1w_generated_train_cor, t1w_generated_train_axial), axis=0)
        t1w_generated_test_all = np.concatenate((t1w_generated_test_sag, 
                                                t1w_generated_test_cor, t1w_generated_test_axial), axis=0)
        
        t1w_std_train_sag = train_batch_output
        t1w_std_train_cor = np.transpose(train_batch_output,[0,2,1,3,4]) 
        t1w_std_train_axial = np.transpose(train_batch_output,[0,3,1,2,4])
        t1w_std_test_sag = test_batch_output
        t1w_std_test_cor = np.transpose(test_batch_output,[0,2,1,3,4])
        t1w_std_test_axial = np.transpose(test_batch_output,[0,3,1,2,4])
        t1w_std_train_all = np.concatenate((t1w_std_train_sag, t1w_std_train_cor, t1w_std_train_axial), axis=0)
        t1w_std_test_all = np.concatenate((t1w_std_test_sag, t1w_std_test_cor, t1w_std_test_axial), axis=0)
        
        t1w_generated_train, t1w_generated_test = t1w_generated_train_all, t1w_generated_test_all
        t1w_std_train, t1w_std_test = t1w_std_train_all, t1w_std_test_all
        
        shape_train = np.shape(t1w_generated_train)
        shape_test = np.shape(t1w_generated_test)
        t1w_generated_train = np.reshape(t1w_generated_train,[shape_train[0]*shape_train[1],shape_train[2],shape_train[3],1])
        t1w_generated_test = np.reshape(t1w_generated_test,[shape_test[0]*shape_test[1],shape_test[2],shape_test[3],1])
        t1w_std_train = np.reshape(t1w_std_train,[shape_train[0]*shape_train[1],shape_train[2],shape_train[3],2])[:,:,:,:input_ch_d]
        t1w_std_test = np.reshape(t1w_std_test,[shape_test[0]*shape_test[1],shape_test[2],shape_test[3],2])[:,:,:,:input_ch_d]
        
        dtrain_input_image_pred = np.zeros(1)
        dtrain_input_image_std = np.zeros(1)
        
        flag1, flag2 = 0, 0
        for jj in range(np.shape(t1w_generated_train)[0]):
            if (np.linalg.norm(t1w_std_train[jj])) > 15:
                flag1 = 1
                if dtrain_input_image_pred.any():
                    dtrain_input_image_pred = np.concatenate((dtrain_input_image_pred, np.expand_dims(t1w_generated_train[jj],0)), axis=0)
                    dtrain_input_image_std = np.concatenate((dtrain_input_image_std, np.expand_dims(t1w_std_train[jj],0)), axis=0)
                else:
                    dtrain_input_image_pred = np.expand_dims(t1w_generated_train[jj],0)
                    dtrain_input_image_std = np.expand_dims(t1w_std_train[jj],0)
    
        dtest_input_image_pred = np.zeros(1)
        dtest_input_image_std = np.zeros(1)    

        for jj in range(np.shape(t1w_generated_test)[0]):
            if (np.linalg.norm(t1w_std_test[jj])) > 15:
                flag2 = 1
                if dtest_input_image_pred.any():
                    dtest_input_image_pred = np.concatenate((dtest_input_image_pred, np.expand_dims(t1w_generated_test[jj],0)), axis=0)
                    dtest_input_image_std = np.concatenate((dtest_input_image_std, np.expand_dims(t1w_std_test[jj],0)), axis=0)
                else:
                    dtest_input_image_pred = np.expand_dims(t1w_generated_test[jj],0)
                    dtest_input_image_std = np.expand_dims(t1w_std_test[jj],0)
        
        doutput_false_train_tag = np.zeros((1,np.shape(dtrain_input_image_pred)[0]))[0] 
        doutput_true_train_tag = np.ones((1,np.shape(dtrain_input_image_std)[0]))[0] * label_smoothing_factor
        doutput_false_test_tag = np.zeros((1,np.shape(dtest_input_image_pred)[0]))[0] 
        doutput_true_test_tag = np.ones((1,np.shape(dtest_input_image_std)[0]))[0] * label_smoothing_factor
        
        dtrain_input_image = np.concatenate((dtrain_input_image_pred, dtrain_input_image_std), axis=0)
        dtrain_output_tag = np.concatenate((doutput_false_train_tag, doutput_true_train_tag), axis=0)
        dtest_input_image = np.concatenate((dtest_input_image_pred, dtest_input_image_std), axis=0)
        dtest_output_tag = np.concatenate((doutput_false_test_tag, doutput_true_test_tag), axis=0)
    
        # train the discriminator
        if (flag1 * flag2):
            print('----------------------------------------------------------------------')
            print('Training the discriminator')
            history1 = model_discriminator.fit(x = dtrain_input_image, 
                                            y = dtrain_output_tag,
                                            validation_data = (dtest_input_image,\
                                                                dtest_output_tag),
                                            batch_size = 10, 
                                            epochs = 3,  
                                            shuffle = True, 
                                            callbacks = None, 
                                            verbose = 2)
    
            model_discriminator.trainable = False
            gtrain_output_tag = np.ones((batch_size_train, block_size[0]*3, 1)) * label_smoothing_factor
            gtest_output_tag = np.ones((batch_size_test, block_size[0]*3, 1)) * label_smoothing_factor
            
            # train the GAN
            print('----------------------------------------------------------------------')
            print('Training the GAN')
            
            history2 = model_gan.fit(x = [train_batch_input, train_batch_bmask], 
                                        y = [train_batch_output, gtrain_output_tag],
                                        validation_data = ([test_batch_input, test_batch_bmask], \
                                                            [test_batch_output, gtest_output_tag]),
                                        batch_size = 1, 
                                        epochs = 1,  
                                        shuffle = True, 
                                        callbacks = None, 
                                        verbose = 2)
            
            l1_loss_train_batch.append(history2.history['model_1_loss'])
            gan_loss_train_batch.append(history2.history['lambda_3_loss'])
            d_loss_train_batch.append(history1.history['loss'])
            l1_loss_test_batch.append(history2.history['val_model_1_loss'])
            gan_loss_test_batch.append(history2.history['val_lambda_3_loss'])
            d_loss_test_batch.append(history1.history['val_loss'])
                                
        cnt_train += batch_size_train
        cnt_test += batch_size_test
        print('Epoch: ', ii + 1)
        
    print('Discriminator loss: train:',np.mean(d_loss_train_batch),'test:', np.mean(d_loss_test_batch))
    print('GAN loss: train:',np.mean(gan_loss_train_batch),'test:', np.mean(gan_loss_test_batch))
    print('l1 loss: train:',np.mean(l1_loss_train_batch),'test:', np.mean(l1_loss_test_batch))
    d_loss_train.append(np.mean(d_loss_train_batch))
    d_loss_test.append(np.mean(d_loss_test_batch))
    gan_loss_train.append(np.mean(gan_loss_train_batch))
    gan_loss_test.append(np.mean(gan_loss_test_batch))
    l1_loss_train.append(np.mean(l1_loss_train_batch))
    l1_loss_test.append(np.mean(l1_loss_test_batch))
    
    # save weights and losses at the end of each epoch
    dpG = os.path.join(dpRoot, 'generator', fnCp)
    if not os.path.exists(dpG):
        os.makedirs(dpG)
    dpD = os.path.join(dpRoot, 'discriminator', fnCp)
    if not os.path.exists(dpD):
        os.makedirs(dpD)
    fpCp1 = os.path.join(dpD, fnCp + '_epoch' + str(ii + 1) + '.h5')
    fpCp2 = os.path.join(dpG, fnCp + '_epoch' + str(ii + 1) + '.h5')
    fpLoss = os.path.join(dpRoot, 'loss', fnCp + '_loss.mat') 
    model_discriminator.save(fpCp1)
    model_generator.save(fpCp2)
    sio.savemat(fpLoss, {'l1_loss_train':l1_loss_train, 'l1_loss_test': l1_loss_test,
                        'gan_loss_train': gan_loss_train, 'gan_loss_test': gan_loss_test,
                        'd_loss_train': d_loss_train, 'd_loss_test': d_loss_test})