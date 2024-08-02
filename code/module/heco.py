import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, alpha, beta):
        super(HeCo, self).__init__()
        print("nei_num:",nei_num) 
        #nei_num: 2
        #feats_dim_list: [1902, 7167, 60]
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        #print("self.fc_list:", self.fc_list[0]) #self.fc_list: Linear(in_features=1902, out_features=128, bias=True)
        # print("self.fc_list[1]:", self.fc_list[1]) #self.fc_list[1]: Linear(in_features=7167, out_features=128, bias=True)
        # print("self.fc_list[2]:", self.fc_list[2]) #self.fc_list[2]: Linear(in_features=60, out_features=128, bias=True)
        #print(len(self.fc_list))#3
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop,tau,lam, beta)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop, tau, alpha)
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.bn1=nn.BatchNorm1d(self.hidden_dim)
        self.bn2=nn.BatchNorm1d(self.hidden_dim)
        self.bn3=nn.BatchNorm1d(self.hidden_dim)
    def forward(self, feats, pos, mps, nei_index):  # p a s
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        # print("len(h_all): ", len(h_all)) #3
        # print("h_all[0].shape: ", h_all[0].shape) #[4019, 128] # acm数据集paper数量
        # print("h_all[1].shape: ", h_all[1].shape) #[7167, 128] # acm数据集author数量
        # print("h_all[2].shape: ", h_all[2].shape) #([60, 128]) # acm数据集subject数量即embedding
        
        z_mp,loss_mp = self.mp(h_all[0], mps)
        # z_mp=self.bn1(z_mp)
        z_sc,loss_sc = self.sc(h_all, nei_index)
        # z_sc=self.bn2(z_sc)
        self.all_embedding=z_mp+z_sc
        loss = self.contrast(z_mp, z_sc, pos)
        return loss+loss_mp+loss_sc

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp,loss_mp = self.mp(z_mp, mps)
        return z_mp.detach()
        # return self.all_embedding.detach()
