from torch.utils.data import Dataset
import os,sys
sys.path.append('../')
import settings
import torch
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from imgaug import augmenters as iaa
from sklearn.model_selection import KFold
config=settings.config

np.random.seed(8)


def overSampling(train_df:pd.DataFrame):
    train_df_orig=train_df.copy()
    lows = [15,15,15,8,9,10,15,8,9,27,10,8,9,10,17,15,20,24,15,26,15,27,15,20,24,17,8,15,27,27,15,27]
    for i in lows:
        target = str(i)
        indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
        train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target+" ")].index
        train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" "+target)].index
        train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
        indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" "+target+" ")].index
        train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)

    return train_df



def read_train_dataset_info():
    path_to_train = config['path_to_train']
    data = pd.read_csv(config['train_csv'])

    data=overSampling(data)

    train_dataset_info = []
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        train_dataset_info.append({
            'path': os.path.join(path_to_train, name),
            'labels': np.array([int(label) for label in labels])})

    print("the kaggle data is ",len(train_dataset_info))

    extra_path=config['extra_data']
    extra_csv=pd.read_csv(config['extra_csv'])

    extra_csv=overSampling(extra_csv)

    for name,labels in zip(extra_csv['Id'],extra_csv['Target'].str.split(' ')):
        path=os.path.join(extra_path,name)
        if os.path.exists(path+"_red.png") and os.path.exists(path+"_yellow.png") \
                and os.path.exists(path+"_green.png") and os.path.exists(path+"_blue.png"):
            train_dataset_info.append({
                'path':os.path.join(extra_path,name),
                'labels':np.array([int(label) for label in labels ])
            })

    print("the kaggle + extra data is ",len(train_dataset_info))
    train_dataset_info = np.array(train_dataset_info)

    return train_dataset_info


def statistic_class_imbalance():
    train_data_info=read_train_dataset_info()
    labels=np.zeros((28,),dtype=np.int32)
    for data in train_data_info:
        labels[data['labels']]+=1


    kf=KFold(n_splits=5,random_state=8,shuffle=True)
    last_train=[]
    for train,test in kf.split(train_data_info):
        print(train,len(train),np.all(train==last_train))
        last_train=train.copy()



class DataLoad_protein(Dataset):

    def __init__(self,dataset_info=None, batch_size=None, shape=None, augument=True):
        super(DataLoad_protein, self).__init__()
        assert shape[2]==4
        self.dataset_info=dataset_info
        self.batch_size=batch_size
        self.shape=shape
        self.augument=augument

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, item):
        if item>len(self.dataset_info):
            item%=len(self.dataset_info)

        while True:
            path=self.dataset_info[item]['path']
            if os.path.exists(path+"_red.png") and os.path.exists(path+"_yellow.png") and os.path.exists(path+"_green.png") and os.path.exists(path+"_blue.png"):
                image=self.load_image(path,self.shape)
                break
            else:
                print("the image is not exist",path)
                item+=1
                item%=len(self.dataset_info)


        if self.augument:
            image = self.augment(image)

        label=np.zeros(shape=(28,))
        label[self.dataset_info[item]['labels']]=1

        image=image/255.#归一化
        image=np.transpose(image,(2,0,1)).copy()

        return torch.from_numpy(image),torch.FloatTensor(label)

    def load_image_extra_data(self,path,shape):
        image=Image.open("path")
        print(np.array(image.shape))

    def load_image(self,path,shape):
        image_red_ch = Image.open(path + '_red.png')
        image_yellow_ch = Image.open(path + '_yellow.png')
        image_green_ch = Image.open(path + '_green.png')
        image_blue_ch = Image.open(path + '_blue.png')
        image = np.stack((
            np.array(image_red_ch),
            np.array(image_green_ch),
            np.array(image_blue_ch),
            np.array(image_yellow_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def augment(self,image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug


if __name__ == '__main__':
    statistic_class_imbalance()