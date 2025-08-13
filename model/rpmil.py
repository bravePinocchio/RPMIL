import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



class RPMIL(nn.Module):
    def __init__(self,args):
        super(RPMIL,self).__init__()

        self.num_heads = 3
        self.D = args.feat_size
        self.n = args.num_sample

        self.attention_V = nn.Sequential(
            nn.Linear(self.D, self.D), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.D, self.D), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.D, 1)
        
        self.fullyconnect1 = nn.Linear(self.D,self.D//8)
        self.fullyconnect21 = nn.Linear(self.D//8,self.D//16)
        self.fullyconnect22 = nn.Linear(self.D//8,self.D//16)
        self.fullyconnect3 = nn.Linear(self.D//16,self.D//8)
        self.fullyconnect4 = nn.Linear(self.D//8,self.D)

        self.norm = nn.LayerNorm(self.D)

        self.crossattQ = nn.Linear(self.D,self.num_heads*self.D,bias=False)
        self.crossattK = nn.Linear(self.D,self.num_heads*self.D,bias=False)
        self.crossattV = nn.Linear(self.D,self.num_heads*self.D,bias=False)
        self.crossO = nn.Linear(self.num_heads*self.D,self.D)
        
        self.head = nn.Linear(self.D,2)
        
        
    def GatedAttention(self,x):
        
        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        
        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        return Z 

    def encode(self,x):

        x = F.relu(self.fullyconnect1(x))
        mu = self.fullyconnect21(x) 
        logvar = self.fullyconnect22(x)

        return mu,logvar
    
    def reparametrisation(self,mu,logvar):
        
        v = torch.exp(logvar * 0.5)
        s = torch.normal(0,1,size=v.shape).cuda()
        z = mu + v.mul(s)
        
        return z
    
    def decode(self,z):

        x_ = self.fullyconnect4(F.relu(self.fullyconnect3(z)))

        return x_
    
    def forward(self,X,args):
        
        x = self.GatedAttention(X)
        
        mu,logvar = self.encode(x)
        
        # resample
        for i in range(self.n):

            if i == 0:
                z = self.reparametrisation(mu,logvar)
            else:
                z = torch.cat((z,self.reparametrisation(mu,logvar)),dim=0)

        x_ = self.decode(z)
        
        #cross att
        Q = self.crossattQ(self.norm(x_)).view(x_.shape[0], self.num_heads, self.D).permute(1, 0, 2).contiguous()
        K = self.crossattK(self.norm(X)).view(X.shape[0], self.num_heads, self.D).permute(1, 2, 0).contiguous()
        V = self.crossattV(self.norm(X)).view(X.shape[0], self.num_heads, self.D).permute(1, 0, 2).contiguous()
        W = F.softmax((Q @ K)/(self.D**0.5),dim=-1)
        h = (W @ V).permute(1, 0, 2).contiguous().view(x_.shape[0], -1)

        h = self.crossO(h)
        
        y_hat = self.head(torch.mean(h,dim=0))
        
    
        return x.squeeze(),torch.mean(x_,dim=0),y_hat,mu,logvar,z
        
