import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .contrast import Contrast
# from .contrast_1 import Contrast2
from .SimCLR_loss import NTXentLoss
from torch.nn import functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp


class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop,tau,lam, beta):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([GCN(hidden_dim, hidden_dim) for _ in range(P)])
        self.att = Attention(hidden_dim, attn_drop)
        # self.m_n_contrast = Contrast(hidden_dim, tau, lam)
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.beta = beta
    def contrastive_loss(self,anchor,positive,negative,margin=0.1):
        anchor=anchor.expand(positive.shape[0],-1)
        negative=negative[:positive.shape[0]]
        distance_positive=F.pairwise_distance(anchor,positive,keepdim=True).pow(2)
        distance_negative=F.pairwise_distance(anchor,negative,keepdim=True).pow(2)
        contrastive_loss=torch.mean((distance_positive-distance_negative+margin).clamp(min=0))
        return contrastive_loss
    def forward(self, h, mps):
        #获取
        embeds = []
        for i in range(self.P):
            embeds.append(self.node_level[i](h, mps[i]))
        loss=0
        # 元路径内对比
        for index,i in tqdm(enumerate(range(self.P))):
            mask=mps[i].to_dense()>0
            for neigh in range(mask.shape[0]):
                neigh_index=torch.nonzero(mask[neigh])  #邻居节点，相似节点
                not_neigh_index=torch.nonzero(~mask[neigh]) #非邻居节点，不相似节点
                loss+=self.contrastive_loss(embeds[i][neigh],embeds[i][neigh_index].squeeze(dim=1),embeds[i][not_neigh_index].squeeze(dim=1))

        # 元路径间对比
        for i in range(len(embeds) - 1):
            if i==0:
                for j in range(i + 1, len(embeds)):
                    self.loss = self.contrast(embeds[i], embeds[j], mps[i])
            else:
                for j in range(i + 1, len(embeds)):
                    self.loss = self.loss + self.contrast(embeds[i], embeds[j], mps[i])

        z_mp = self.att(embeds)
        #聚合后的特征
        return z_mp, loss+self.beta*self.loss


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def criterion(out_1, out_2, tau_plus, batch_size, beta, estimator,tau):
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / tau)
    mask = get_negative_mask(batch_size)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / tau)
    pos = torch.cat([pos, pos], dim=0)

    # negative samples similarity scoring
    if estimator == 'hard':
        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / tau))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng))).mean()

    return loss