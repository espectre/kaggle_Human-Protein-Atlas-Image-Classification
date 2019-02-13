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
from torch.nn import DataParallel
from torch.autograd import Variable
import warnings
from sklearn.model_selection import train_test_split,KFold
from model import DenseNet121
from utils import *
from loss import *
from Data import *
config=settings.config

#os.environ["CUDA_VISIBLE_DEVICES"]=config["CUDA_VISIBLE_DEVICES"]


def main():

    print(config)
    torch.manual_seed(0)

    train_dataset_info=read_train_dataset_info()
    indexes=np.arange(train_dataset_info.shape[0])
    np.random.shuffle(indexes)
    train_indexes, valid_indexes = train_test_split(indexes, test_size=0.1, random_state=8)
    train_loader,validation_loader,net,loss,opt,start_epoch=\
            build_network(train_dataset_info[indexes],train_dataset_info[indexes])

    net.cuda()
    net=DataParallel(net)
    loss.cuda()

    for epoch in range(start_epoch+1, start_epoch+config['epochs']):
        print("the epoch is %d"%epoch)
        train(train_loader,net,loss,epoch,opt,get_lr,config['save_freq'],config['save_dir'])#当继续训练的时候，这个时候学习率有变化
        validation(validation_loader,net,loss,epoch,config['config'])


def main2():
    '''
    for K-fold cross-validation
    :return:
    '''
    print(config)
    torch.manual_seed(0)

    train_dataset_info=read_train_dataset_info()
    kf=KFold(n_splits=5,random_state=8,shuffle=True)

    fold=0
    os.environ['CUDA_VISIBLE_DEVICES']=config['n_gpu']

    for train_data,test_data in kf.split(train_dataset_info):

        fold+=1

        train_loader,validation_loader,net,loss,opt,start_epoch=\
            build_network(train_dataset_info[train_data],train_dataset_info[test_data])

        net.cuda()
        net=DataParallel(net)
        loss.cuda()

        save_dir=os.path.join(config['save_dir'],"_"+str(fold)+"fold")

        for epoch in range(start_epoch+1, start_epoch+config['epochs']):
            print("the epoch is %d"%epoch)
            train(train_loader,net,loss,epoch,opt,get_lr,config['save_freq'],save_dir)#当继续训练的时候，这个时候学习率有变化
            validation(validation_loader,net,loss,epoch,save_dir)


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir,visual=None):
    net.train()
    lr=get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

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

        output=output>0 #最后一层没有sigmoid激活，所以要大于0
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

    if epoch%save_freq==0:
        #-----------for training restart---------------
        state_dict=net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key]=state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict},
            os.path.join(save_dir, 'train_%d_acc_%f.ckpt' % (epoch,acc)))

min_loss=np.inf
def validation(data_loader, net, loss, epoch, save_dir,visual=None):
    global min_loss
    net.eval()

    metrics=[]
    loss_value=0
    count=0
    for i, (data,label) in enumerate(data_loader):
        data=Variable(data.cuda(async=True))
        label=Variable(label.cuda(async=True))
        output=net(data.float())
        loss_output=loss(output,label)
        loss_value+=loss_output.item()
        print("the loss is",loss_output.item())
        count+=1
        output=output>0
        label=label>0
        for j in range(data.size()[0]):
            metrics.append(torch.equal(output[j],label[j]))


    metrics=np.array(metrics)*1.0

    acc=metrics.sum()/metrics.shape[0]


    print("------------------------------------------------")
    print("the epoch is %d, and the validation acc is %f"%(epoch,acc))
    print("the loss value is ",loss_value)
    print("------------------------------------------------")

    if min_loss>loss_value:
        #-------save the best-------------
        min_loss=loss_value
        state_dict=net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key]=state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict},
            os.path.join(save_dir, 'the_min_loss_%f_epoch_%d_acc_%f.ckpt' % (min_loss,epoch,acc)))

if __name__ == '__main__':
    if config['Kfold']:
        print("KFold")
        main2()
    else:
        config['batch_size']=12
        main()







