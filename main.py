import torch,torchvision
from torch import nn
import torch.nn.functional as F
from torch import optim
from model.rpmil import RPMIL
import numpy as np
import os,argparse

import random
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score,accuracy_score, precision_score, f1_score
from DataLoader import Camelyondataset,TCGAdataset
from torch.utils.data import DataLoader

def main():

    parser = argparse.ArgumentParser(description='Train rpmil on 20x patch features learned by resnet 50')
    parser.add_argument('--seed', default=2024, type=int, help='random seed')
    parser.add_argument('--r', default=[0.5,0.5,1], type=list, help='r0 -> mse, r1 -> kl, r2 -> bce')
    parser.add_argument('--feat_size', default=1024, type=int, help='feats_size')
    parser.add_argument('--lr', default=0.0001,type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='tcga', type=str, help='Dataset folder name: c16 or tcga')
    parser.add_argument('--num_sample', default=1000, type=int, help='number of sampling')
    parser.add_argument('--model', default='RPMIL', type=str, help='MIL model [dsmil]')
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    
    if args.dataset == 'c16':

        train_path = pd.read_csv("./datasets/c16_train.csv")
        test_path = pd.read_csv("./datasets/c16_test.csv")
        trainset =  Camelyondataset(train_path)
        train_loader = DataLoader(trainset,1, shuffle=True, num_workers=1)
        testset =  Camelyondataset(test_path)
        test_loader = DataLoader(testset,1, shuffle=False, num_workers=1)

    elif args.dataset == 'tcga':
        
        bags_csv = shuffle(pd.read_csv('./datasets/tcga_nsclc.csv')).reset_index(drop=True) 
        train_path = bags_csv[:650]
        val_path = bags_csv[650:800]
        test_path = bags_csv[800:]
        trainset =  TCGAdataset(train_path)
        train_loader = DataLoader(trainset,1, shuffle=True, num_workers=1)
        valset =  TCGAdataset(val_path)
        val_loader = DataLoader(valset,1, shuffle=True, num_workers=1)
        testset =  TCGAdataset(test_path)
        test_loader = DataLoader(testset,1, shuffle=False, num_workers=1)

    train_bags = train_loader

    if args.dataset == 'tcga':
        test_bags = val_loader
    else:
        test_bags = test_loader
    
    model = RPMIL(args).cuda()
    print(model)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    for epoch in range(args.num_epochs):
        print('====> Epoch: {} '.format(epoch))
        train(train_bags,model,optimizer,args)
        test(test_bags,model,args)
        scheduler.step()
        
    if args.dataset == 'tcga':
        test_bags = test_loader
        test(test_bags,model,args)


MSEloss = nn.MSELoss(reduction='sum')
BCEloss = nn.CrossEntropyLoss()   
    

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_function(recon_x, x, y_hat,y, mu, logvar):

    """
    recon_x: generating x
    x: origin x
    y_hat: predicating label
    y: true label
    mu: latent mean
    logvar: latent log variance
    """
    
    BCE = BCEloss(y_hat,y)

    MSE = MSEloss(recon_x,x.detach())
    
    KLD_element = torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar
    KLD = 0.5 * torch.sum(KLD_element)
    
    return MSE,KLD,BCE


    
def train(train_path,model,optimizer,args):
    # train
    train_labels = []
    train_predictions = []
    y_true = []
    y_pred = []
    model.train()
    train_mse_loss = 0
    train_kl_loss = 0
    train_bce_loss = 0
    train_total_loss = 0
    train_length = len(train_path)
    
    for i,(label,feats) in enumerate(train_path):
        optimizer.zero_grad()

        bag_label = F.one_hot(torch.tensor(int(label)),2).squeeze().float().cuda()
        bag_feats = feats.view(-1, args.feat_size).float().cuda()
        
        x, x_, y_hat,mu, logvar,z= model(bag_feats,args)
        
        mse_loss,kl_loss,bce_loss = loss_function(x_, x, y_hat, bag_label, mu, logvar)
        
        loss = args.r[0]*mse_loss + args.r[1]*kl_loss + args.r[2]*bce_loss
        loss.backward()
        
        train_total_loss += mse_loss.item() + kl_loss.item() + bce_loss.item()
        train_mse_loss += mse_loss.item()
        train_kl_loss += kl_loss.item()
        train_bce_loss += bce_loss.item()
        optimizer.step()
        
        
        if int(label[0]) == 0:
            y_true.append(0)                       
        else:
            y_true.append(1)
        if torch.argmax(y_hat)==0 :
            y_pred.append(0)
        else:
            y_pred.append(1)
            
        train_labels.extend(label)
        bag_prediction = F.softmax(y_hat.detach(), dim=-1)[1].reshape(1)
        train_predictions.extend(bag_prediction.cpu().numpy())

    
    print('Train - Average total loss: {:.4f} mse_loss: {:.4f} kl_loss: {:.4f} bce_loss: {:.4f} acc: {:.4f} p: {:.4f} f1: {:.4f} auc: {:.4f}'.format((train_total_loss)/train_length, 
                                                                                                                                                     (train_mse_loss)/train_length,
                                                                                                                                                     (train_kl_loss)/train_length,
                                                                                                                                                     (train_bce_loss)/train_length,
                                                                                                                                                     accuracy_score(y_true, y_pred), 
                                                                                                                                                     precision_score(y_true, y_pred),
                                                                                                                                                     f1_score(y_true, y_pred),
                                                                                                                                                     roc_auc_score(train_labels, train_predictions)))
    
def test(test_path,model,args):
    #test
    v_test=[]
    test_labels = []
    test_predictions = []
    y_true = []
    y_pred = []
    test_mse_loss = 0
    test_kl_loss = 0
    test_bce_loss = 0
    test_length = len(test_path)
    model.eval()
    
    with torch.no_grad():
        test_loss = 0
        for i,(label,feats) in enumerate(test_path):
            
            bag_label = F.one_hot(torch.tensor(int(label)),2).squeeze().float().cuda()
            bag_feats = feats.view(-1, args.feat_size).float().cuda()
            
            x, x_, y_hat,mu, logvar,z= model(bag_feats,args)
            
            mse_loss,kl_loss,bce_loss = loss_function(x_, x, y_hat, bag_label, mu, logvar)

            test_mse_loss += mse_loss.item()
            test_kl_loss += kl_loss.item()
            test_bce_loss += bce_loss.item()
            
            if int(label[0]) == 0:
                y_true.append(0)                       
            else:
                y_true.append(1)
            if torch.argmax(y_hat)==0:
                y_pred.append(0)
            else:
                y_pred.append(1)
            
            test_labels.extend(label)
            bag_prediction = F.softmax(y_hat, dim=-1)[1].reshape(1)
            test_predictions.extend(bag_prediction.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        test_total_loss = test_mse_loss + test_kl_loss + test_bce_loss
        print('Test - Average total loss: {:.4f} mse_loss: {:.4f} kl_loss: {:.4f} bce_loss: {:.4f} acc: {:.4f} p: {:.4f} f1: {:.4f} auc: {:.4f}'.format(test_total_loss/test_length,
                                                                                                                                                        (test_mse_loss)/test_length,
                                                                                                                                                        (test_kl_loss)/test_length,
                                                                                                                                                        (test_bce_loss)/test_length,
                                                                                                                                                        accuracy_score(y_true, y_pred),
                                                                                                                                                        precision_score(y_true, y_pred,average='weighted'),
                                                                                                                                                        f1_score(y_true, y_pred),
                                                                                                                                                        roc_auc_score(test_labels, test_predictions)))


if __name__ == '__main__':
    
    main()
    


