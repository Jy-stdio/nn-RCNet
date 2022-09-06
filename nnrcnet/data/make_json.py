'''
参考csdn https://blog.csdn.net/u014776464/article/details/109036624
官方文档 https://blog.csdn.net/u014776464/article/details/109036624

'''

from email.mime import image
import glob
import os
import re
import json
from collections import OrderedDict

path_originalData = "/home3/HWGroup/liujy/rectal_cancer/rectal_cancer_segmentation/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/"

def list_sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """   
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)
    return l


def remove_miss_gt(images, images1, images2):
    """只保留有三类gt全的图像
    """
    ids = [i.split("/")[-1].split('O')[0] for i in images]
    ids1 = [i.split("/")[-1].split('O')[0] for i in images1]
    ids2 = [i.split("/")[-1].split('O')[0] for i in images2]

    new1 = [i for i in ids if i in ids1]
    new2 = [i for i in new1 if i in ids2]
    new_images = [os.path.join(path_originalData,"Task58_Rectal-C/imagesTr/",i+'O.nii.gz') for i in new2]
    labels_C = [os.path.join(path_originalData,"Task58_Rectal-C/labelsTr/",i+'C-label.nii.gz') for i in new2]
    labels_H = [os.path.join(path_originalData,"Task62_Rectal-H/labelsTr/",i+'H-label.nii.gz') for i in new2]
    labels_T = [os.path.join(path_originalData,"Task65_Rectal-T/labelsTr/",i+'TL-label.nii.gz') for i in new2]

    return new_images, labels_C, labels_H, labels_T


images = list_sort_nicely(glob.glob(path_originalData+"Task58_Rectal-C/imagesTr/*"))
images1 = list_sort_nicely(glob.glob(path_originalData+"Task62_Rectal-H/imagesTr/*"))
images2 = list_sort_nicely(glob.glob(path_originalData+"Task65_Rectal-T/imagesTr/*"))

new_images, labels_C, labels_H, labels_T = remove_miss_gt(images, images1, images2)

train_ratio = 0.7
datasize = len(new_images)
train_size = int(train_ratio * datasize)
val_size = datasize - train_size


#####下面是创建json文件的内容
#可以根据你的数据集，修改里面的描述
json_dict = OrderedDict()
json_dict['name'] = "rectal cancer"
json_dict['description'] = " Segmentation"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "see challenge website"
json_dict['licence'] = "see challenge website"
json_dict['release'] = "0.0"
#这里填入模态信息，0表示只有一个模态，还可以加入“1”：“MRI”之类的描述，详情请参考官方源码给出的示例
json_dict['modality'] = {
    "0": "T2WI"
}

#这里为label文件中的多个标签，比如这里有血管、胆管、结石、肿块四个标签，名字可以按需要命名
json_dict['labels'] = {
    "0": "Background",
    "1": "label",  # label

}

#下面部分不需要修改>>>>>>
json_dict['numTraining'] = train_size
json_dict['numVal'] = val_size
# print(datasize)

json_dict['training'] = []
json_dict['val'] = []
for idx in range(train_size):
    json_dict['training'].append({'image': f"{new_images[idx]}", "label_C": f"{labels_C[idx]}", "label_H": f"{labels_H[idx]}", "label_T": f"{labels_T[idx]}"})

for idx in range(val_size):
    # print(train_size+idx)
    json_dict['val'].append({'image': f"{new_images[train_size+idx]}", "label_C": f"{labels_C[train_size+idx]}", "label_H": f"{labels_H[train_size+idx]}", "label_T": f"{labels_T[train_size+idx]}"})


with open(os.path.join("./dataset/dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)

