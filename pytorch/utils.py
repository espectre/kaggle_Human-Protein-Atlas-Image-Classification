import torch
from collections import OrderedDict
import numpy as np
import sys
import os
sys.path.append("../")
sys.path.append("./")
import settings
from Data import DataLoad_protein
from torch.utils.data import DataLoader
from model import DenseNet121
from torch.nn import BCEWithLogitsLoss,BCELoss


config=settings.config_linux

def swa(save_name):
    new_model=OrderedDict()
    model_number=len(config['swa_list'])
    new_model=torch.load(os.path.join(config['model_path_root'],config['swa_list'][0]))['state_dict'].copy()
    for i in range(1,model_number):
        model=torch.load(os.path.join(config['model_path_root'],config['swa_list'][i]))['state_dict']

        for key in model.keys():
            new_model[key]+=model[key]


    for key in new_model.keys():
        model=torch.load(os.path.join(config['model_path_root'],config['swa_list'][0]))['state_dict']
        print("before ",key,new_model[key],"model1",model[key])
        new_model[key]/=model_number
        print("After ",key,new_model[key])

    print("done")

    torch.save({'state_dict':new_model},save_name)


def get_lr(x):

    lr = config['lr']
    epochs = config['epochs']
    return lr * (np.cos(np.pi * x / epochs) + 1.) / 2


def get_lr2(epoch):
    if epoch <= config['epochs'] * 0.3:
        lr = config['lr']
    elif epoch <= config['epochs'] * 0.6:
        lr = 0.1 * config['lr']
    else:
        lr = 0.01 * config['lr']
    return lr

def build_network(train_dataset,val_dataset):

    train_dataset=DataLoad_protein(train_dataset,config['batch_size'],(config['SIZE'],config['SIZE'],4),
                                   augument=True)
    val_dataset=DataLoad_protein(val_dataset,config['batch_size'],(config['SIZE'],config['SIZE'],4),
                                 augument=False)



    train_loader=DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=config['batch_size'],
        pin_memory=True,
        num_workers=config['num_workers'],
    )
    validation_loader=DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=3,
        pin_memory=True,
        num_workers=config['num_workers']
    )

    net=DenseNet121(pretrained=True)

    start_epoch=config['start_epoch']

    print("load the fine model",config['finetune_model'])
    fine_model=torch.load(config['finetune_model'])
    net.load_state_dict(fine_model['state_dict'])

    if config['resume'] is not None:
        model=torch.load(config['resume'])
        start_epoch=model['epoch']
        net.load_state_dict(model['state_dict'])

    opt=torch.optim.Adam(
        net.parameters(),
        lr=config['lr']
    )#without

    loss=BCEWithLogitsLoss()

    return train_loader,validation_loader,net,loss,opt,start_epoch



if __name__ == '__main__':
    #swa(os.path.join(config['model_path_root'],"swa_weight.ckpt"))
    for epoch in range(60):
        print(get_lr(epoch))
