# s_DeepAnat_applyCNN.py
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
from tensorflow.keras.models import load_model

import aini_utils as utils

# %% set up path
dpRoot = os.path.dirname(os.path.abspath('s_DeepAnat_applyCNN.py'))
os.chdir(dpRoot)

# %% subjects
dpData = '/autofs/space/rhapsody_001/users/qtian/AINI/DeepAnat'
subjects = sorted(glob.glob(os.path.join(dpData, 'evaluation_subjects', 'Tract*')))

# %% load network
fnCp = 'unet_all2t1'
print(fnCp)

fpCp = os.path.join(dpRoot, fnCp, fnCp + '.h5') 
model_unet = load_model(fpCp, custom_objects={'mean_absolute_error_weighted': utils.mean_absolute_error_weighted})
            
# %% load data
sz_block = 64
sz_pad = 5
sz_crop = 3
input_list = ['diff_meanb0', 'diff_meandwi', 'diff_dtiL1', 'diff_dtiL2', 'diff_dtiL3',
              'diff_dtiDwi1', 'diff_dtiDwi2', 'diff_dtiDwi3', 'diff_dtiDwi4', 'diff_dtiDwi5', 'diff_dtiDwi6']

mse = []

for ii in np.arange(len(subjects)):
    sj = os.path.basename(subjects[ii])
    
    print(sj)
    dpSub = os.path.join(dpData, 'evaluation_subjects', sj)
    
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

    norm_ch = [0, 1, 5, 6, 7, 8, 9, 10] # do not normalize DTI metrics
    t1w_norm, tmp = utils.normalize_image(t1w, t1w, mask)
    inputs_norm, tmp = utils.normalize_image(inputs, inputs, mask, norm_ch) 
    
    ind_block, ind_brain = utils.block_ind(mask, sz_block=sz_block, sz_pad=sz_pad)
    inputs_norm_block = utils.extract_block(inputs_norm, ind_block)
    mask_block = utils.extract_block(mask, ind_block)
    
    t1w_pred_block = np.zeros(mask_block.shape)
    
    for mm in np.arange(0, mask_block.shape[0]):
        tmp = model_unet.predict([inputs_norm_block[mm:mm+1, :, :, :, :], mask_block[mm:mm+1, :, :, :, :]]) 
        t1w_pred_block[mm:mm+1, :, :, :, :] = tmp[:, :, :, :, :-1]

    t1w_pred_vol, tmp = utils.block2brain(t1w_pred_block, ind_block, mask, sz_crop)

    fpPred = os.path.join(dpSub, sj + fnCp + '_predimg_norm.nii.gz')
    utils.save_nii(fpPred, t1w_pred_vol, fpMask)
    
    t1w_mse = (t1w_norm + 3) / 6
    pred_mse = (t1w_pred_vol + 3) / 6
    mse_subject = np.mean((t1w_mse[mask > 0.5] - pred_mse[mask > 0.5]) ** 2)
    print('mean squared error:', mse_subject)
    mse.append(mse_subject)
    
    # transform standardized intensities to normal range
    # can use the mean and std from one of training subjects
    img_mean = np.mean(t1w[mask > 0.5])
    img_std = np.std(t1w[mask > 0.5])
    
    t1w_pred_final = (t1w_pred_vol * img_std + img_mean) * mask
    fpPred = os.path.join(dpSub, sj + fnCp + '_predimg_final.nii.gz')
    utils.save_nii(fpPred, t1w_pred_final, fpMask)

    



















