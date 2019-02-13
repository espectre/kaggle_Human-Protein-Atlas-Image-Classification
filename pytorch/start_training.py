import os, sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn import BCEWithLogitsLoss,BCELoss
from torch.nn.functional import sigmoid
from sklearn.utils import class_weight, shuffle
sys.path.append('../')
sys.path.append('./')
import torch
import settings
from torch.autograd import Variable
import warnings
from sklearn.model_selection import train_test_split
from model import DenseNet121
from Data import *
config=settings.config
os.environ["CUDA_VISIBLE_DEVICES"]=config["CUDA_VISIBLE_DEVICES"]


def main():
    torch.manual_seed(0)
    train_dataset_info=read_train_dataset_info()
    indexes=np.arange(train_dataset_info.shape[0])
    np.random.shuffle(indexes)
    train_indexes, valid_indexes = train_test_split(indexes, test_size=0.1, random_state=8)
    train_dataset=DataLoad_protein(train_dataset_info[train_indexes],config['batch_size'],(config['SIZE'],config['SIZE'],4),
                                   augument=True)
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        pin_memory=True,
        num_workers=config['num_workers'],
    )
    net=DenseNet121(pretrained=True,start_first="start_training")

    start_epoch=config['start_epoch']

    if config['resume'] is not None:
        print("load the model",config['resume'])
        model=torch.load(config['resume'])
        start_epoch=model['epoch']
        net.load_state_dict(model['state_dict'])

    net.cuda()
    first_conv=net.state_dict()['base_model.features.denseblock3.denselayer22.conv2.weight'].clone()
    second_classifier=net.state_dict()['S.conv1.weight'].clone()
    second_separ=net.state_dict()['classifier.weight'].clone()




    loss=BCEWithLogitsLoss()
    opt=torch.optim.Adam(
        filter(lambda p:p.requires_grad,net.parameters()),
        lr=1e-3,
    )

    for epoch in range(start_epoch+1,start_epoch+6):
        print("epoch is ",epoch)
        train(train_loader,net,loss,epoch,opt,config['save_freq'],config['save_dir'])

    first_conv_after=net.state_dict()['base_model.features.denseblock3.denselayer22.conv2.weight'].clone()
    second_classifier_after=net.state_dict()['S.conv1.weight'].clone()
    second_separ_after=net.state_dict()['classifier.weight'].clone()
    print(net.state_dict()['base_model.features.denseblock3.denselayer22.conv2.weight'])
    print("the training parameters",net.state_dict()['S.conv1.weight'],net.state_dict()['classifier.weight'])
    print(torch.equal(first_conv_after,first_conv))
    print(torch.equal(second_classifier,second_classifier_after))
    print(torch.equal(second_separ,second_separ_after))


def train(data_loader, net, loss, epoch, optimizer, save_freq, save_dir,visual=None):
    net.train()
    metrics=[]
    for i, (data,label) in enumerate(data_loader):
        data=Variable(data.cuda(async=True))
        label=Variable(label.cuda(async=True))

        output=net(data.float())
        loss_output=loss(output,label)
        print("the loss is",loss_output.data)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        output=output>0
        label=label>0
        for j in range(data.size()[0]):
            metrics.append(torch.equal(output[j],label[j]))
    metrics=np.array(metrics)*1.0

    acc=metrics.sum()/metrics.shape[0]
    print("------------------------------------------------")
    print("the epoch is %d, and the acc is %f"%(epoch,acc))
    print("------------------------------------------------")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        #-----------for training restart---------------
    state_dict=net.state_dict()
    for key in state_dict.keys():
        state_dict[key]=state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'save_dir': save_dir,
        'state_dict': state_dict},
        os.path.join(save_dir, '%d_%f.ckpt' % (epoch,acc)))



if __name__ == '__main__':
    main()