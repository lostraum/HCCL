import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SimCLR_loss import NTXentLoss


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
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
        print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb


class Sc_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop, tau, alpha):
        super(Sc_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.nl = NTXentLoss(tau)
        self.nl_intra = NTXentLoss(tau)
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.alpha = alpha

    def forward(self, nei_h, nei_index):
        embeds = []
        loss = 0.
        loss_intra = 0.
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
                                                               replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda()
            one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
            embeds.append(one_type_emb)
            # print("one_type_emb的类型",type(one_type_emb))
            # print("one_type_emb的大小", one_type_emb.shape)
            # print("nei_h的类型",type(nei_h)) # list
            # print("nei_h的长度", len(nei_h)) #nei_h的长度 3
            # print("nei_h[0]的大小", nei_h[0].shape) #nei_h[0]的大小 torch.Size([4019, 128])


            #类型内对比
            loss_intra = loss_intra + self.nl_intra(nei_h[0], one_type_emb)# nei_h[0]为目标节点嵌入

        # 类型间对比
        for i in range(len(embeds) - 1):
            for j in range(i + 1, len(embeds)):
                loss = loss + self.nl(embeds[i], embeds[j])
        # loss = loss + self.nl(embeds[0], embeds[1]) #针对acm数据集，只有两种邻居节点
                
        z_mc = self.inter(embeds)
        loss_total = loss_intra+ self.alpha*loss
        return z_mc, loss_total
