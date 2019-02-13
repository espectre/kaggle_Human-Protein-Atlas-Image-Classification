import torch
import torch.nn.functional as F
import numpy as np
import torch.nn
def dice_loss(input,target):
    '''
    make the soft dice loss
    :param input: input
    :param target: mask label
    :return:
    '''
    input=torch.sigmoid(input)
    smooth=1.0#for soft
    flat_input=input.view(-1)
    flag_target=target.view(-1)
    intersection=(flat_input*flag_target).sum()

    return 2*(intersection+smooth)/(flag_target.sum()+flat_input.sum()+smooth)

class FocalLoss(torch.nn.Module):
    def __init__(self,gamma=2,size_average=True):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
        self.size_average=size_average

    def forward(self, input,target,class_weight=None):
        '''
        here are FocalLoss
        :param input:
        :param target:
        :param class_weight:
        :return:
        '''


        target=target.view(-1,1).long()
        if class_weight==None:
            class_weight=[1]*2
        prob=F.sigmoid(input)
        prob=prob.view(-1,1)
        prob=torch.cat((1-prob,prob),1)
        select=torch.FloatTensor(len(prob),2).zero_().cuda()
        select.scatter_(1,target,1)
        class_weight=torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight=torch.gather(class_weight,0,target)
        prob=(prob*select).sum(1).view(-1,1)
        prob=torch.clamp(prob,1e-8,1-1e-8)
        batch_loss=-class_weight*(torch.pow((1-prob),self.gamma))*prob.log()
        if self.size_average:
            loss=batch_loss.mean()

        else:
            loss=batch_loss

        return loss


class MixedLoss(torch.nn.Module):

    def __init__(self,alpha,gamma):
        super(MixedLoss,self).__init__()
        self.alpha=alpha
        self.focal=FocalLoss(gamma=gamma)

    def forward(self, input,target):
        loss=self.alpha*self.focal(input,target)-torch.log(dice_loss(input,target))
        return loss.mean()



class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, size_average=True):
        print("focalLoss")
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)

            prob   = torch.cat((1-prob, prob), 1)

            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)#clip by value
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss

