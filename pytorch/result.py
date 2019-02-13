import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import os,sys
sys.path.append("./")
sys.path.append("../")
import settings
from Data import DataLoad_protein
from model import DenseNet121
from PIL import Image
from imgaug import augmenters as iaa
import cv2
import ntpath
config=settings.config


def load_image(path,shape):
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

def numpy_to_tensor(image):
    '''

    :param image: h,w,c
    :return:
    '''
    image=np.transpose(image,axes=(2,0,1)).copy()
    image=image[np.newaxis]
    return torch.from_numpy(image)

def get_result(threshold=[0.2]):
    submit = pd.read_csv(config['sample_submission'])

    if not os.path.exists("./predict_result2/"):
        os.mkdir("./predict_result2")

    state_dict=torch.load(config['load_model_path'])

    epoch=ntpath.basename(config['load_model_path'])

    net=DenseNet121()
    print(config['load_model_path'])
    net.load_state_dict(state_dict['state_dict'])#load the model
    net.cuda()
    net.eval()
    for thresh in threshold:
        predicted = []
        draw_predict = []
        for name in tqdm(submit['Id']):

            path=os.path.join(config['path_to_test'],name)
            image1 = load_image(path, (config['SIZE'],config['SIZE'], 3))/255.
            image2 = np.fliplr(image1)
            image3 = np.flipud(image1)
            image4 = np.rot90(image1)
            image5 = np.rot90(image4)
            image6 = np.rot90(image5)


            images=np.stack([image1,image2,image3,image4,image5,image6]).transpose(0,3,1,2).copy()
            images=torch.from_numpy(images)
            score_predict=net(images.float().cuda())
            score_predict=F.sigmoid(score_predict)#sigmoid激活
            score_predict=torch.mean(score_predict,dim=0).detach().cpu().numpy()#转成numpy
            draw_predict.append(score_predict)
            label_predict = np.arange(28)[score_predict >= thresh]

            if int(label_predict.shape[0])==0:
                print("the label is None")
                print(score_predict)
                label_predict=np.arange(28)[np.argsort(score_predict)[-2:]]
                print(label_predict)

            str_predict_label = ' '.join(str(l) for l in label_predict)
            predicted.append(str_predict_label)


        submit['Predicted'] = predicted
        np.save('./predict_result2/draw_predict_DenseNet121_512_'+str(thresh)+'_'+str(epoch)+'.npy', score_predict)
        submit.to_csv("./predict_result2/"+str(epoch)+".csv", index=False)



if __name__ == '__main__':
    get_result()