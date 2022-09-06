"""
数据预处理
——————————
返回数据
- 每个样本：(image,mask,label) or (image,mask)
- txt(每行):文件名,是否label{0,1}

----------
TODO:
1. 
"""
from cmath import isnan, nan
import json
import pickle
import os
import numpy as np
import nibabel as nib

import pandas as pd

root = '/home3/HWGroup/liujy/nn-RCNet/nnrcnet/'

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def zscore(image):
    mask = image.sum(-1) > 0
    for k in range(3):

        x = image[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        image[..., k] = x
    return image


def process_f32b0(img_gt, datacsv, output, seg_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    images = np.array(nib_load(img_gt['image']), dtype='float32', order='C')
    if seg_label:
        label_C = np.array(nib_load(img_gt['label_C']), dtype='uint8', order='C')
        label_H = np.array(nib_load(img_gt['label_H']), dtype='uint8', order='C')
        label_T = np.array(nib_load(img_gt['label_T']), dtype='uint8', order='C')
    id = int(img_gt['image'].split('/')[-1].split('-')[0])  # 样本id

    # images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]

    images = zscore(images)

    # 获取label：T/N/EMVI/CRM
    row= datacsv.loc[(datacsv["ID_num"]==id)]
    T = float('nan')
    if row["ID_num"].tolist()!=[]:
        # T, N, EMVI, CRM = row[['T', 'N', 'EMVI', 'CRM']]  # 都为2分类
        T = row.iloc[0].at['T']
    if np.isnan(T):
        T = -1
    f = open(output, 'ab')

    if seg_label:
            pickle.dump((images, label_C, label_H, label_T, T), f)
    else:
        return

def main(datajson, datacsv):
    training = datajson['training']
    val = datajson['val']

    for img_gt in training:
        process_f32b0(img_gt, datacsv, output_train)
    for img_gt in val:
        process_f32b0(img_gt, datacsv, output_val)

if __name__ == '__main__':

    jsonfile = open(os.path.join(root, 'data/dataset/dataset.json'))
    datajson = json.load(jsonfile)

    csvfile = '/home3/HWGroup/liujy/nn-RCNet/Radiomics/data/feature-label.csv'
    datacsv = pd.read_csv(csvfile)
    # T标签转为2分类
    datacsv.loc[datacsv['T']<=2, 'T'] = 0
    datacsv.loc[datacsv['T']>2, 'T'] = 1

    # 训练集和测试集序列化保存
    output_train = os.path.join(root, 'data/dataset/data_f32b0_train.pkl')
    output_val = os.path.join(root, 'data/dataset/data_f32b0_val.pkl')
    
    main(datajson, datacsv)