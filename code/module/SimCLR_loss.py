import torch
import torch.nn as nn
import torch.nn.functional as F
class NTXentLoss(nn.Module):
    def __init__(self,temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature=temperature

    def forward(self, h1, h2, norm=True, temperature=0.3):
        # h1 [N, D] augment of i    [[a1], [a2], [a3]]
        # h2 [N, D] augment of j    [[b1], [b2], [b3]]

        # norm: pre normalize of cos similarity
        if norm:
            h1 = F.normalize(h1, dim=-1)
            h2 = F.normalize(h2, dim=-1)

        # vector to computer   [[a1], [a2], [a3], [b1], [b2], [b3]]
        h = torch.cat([h1, h2], dim=0)

        # [2N, D] @ [D, 2N] -> [2N, 2N]   [[a1a1],[a1a2] ...]
        sim_matrix = torch.exp(torch.mm(h, h.t().contiguous()) / self.temperature)

        # delete the self batch  delete [a1a1, a2a2 ...]
        mask = (torch.ones_like(sim_matrix) - torch.eye(h.size(0), device=sim_matrix.device)).bool()

        # # [2*N, 2*N-1]
        sim_matrix = sim_matrix.masked_select(mask).view(h.size(0), -1)

        # positive sample [a1b1, a2b2, a3b3]   sum都是去掉最后一维的channel
        pos_sim = torch.exp(torch.sum(h1 * h2, dim=-1) / self.temperature)
        # two direction ij and ji
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
        return loss.mean()
