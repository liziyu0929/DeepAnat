# s_DeepAnat_trainUNet.py
#
# (c) Ziyu Li, Qiyuan Tian, Artificial Intelligence in Neuroimaging Software, 2022

# %% load moduals
import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nb
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import aini_utils as utils
from cnn_models import unet_3d_model

# %% set up path
dpRoot = os.path.dirname(os.path.abspath('s_DeepAnat_trainUNet.py'))
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

# %% set up model
num_ch = train_block_in.shape[-1]
num_ker = 48
model_unet = unet_3d_model(num_ch=num_ch, filter_num=num_ker)
model_unet.summary()

# %% set up optimizer
adam_opt_unet = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model_unet.compile(loss = utils.mean_absolute_error_weighted, optimizer = adam_opt_unet)

# %% train
sz_batch = 1
num_epochs = 30

fnCp = 'unet_all2t1'
dpCnn = os.path.join(dpRoot, fnCp) 
if not os.path.exists(dpCnn):
    os.mkdir(dpCnn)
    print('create directory')
        
fpCp = os.path.join(dpCnn, fnCp + '.h5')
cp = ModelCheckpoint(fpCp, monitor='val_loss', save_best_only = True)

history = model_unet.fit(x = [train_block_in, train_block_mask], 
                         y = train_block_out, 
                         validation_data = ([valid_block_in, valid_block_mask], valid_block_out),
                         batch_size = sz_batch, 
                         epochs = num_epochs,  
                         shuffle = True, 
                         callbacks = [cp], 
                         verbose = 1)

fpLoss = os.path.join(dpCnn, fnCp + '.mat') 
sio.savemat(fpLoss, {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})    
            





