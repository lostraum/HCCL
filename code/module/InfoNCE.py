import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCEContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCEContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output1, output2):
        outputs = torch.cat([output1.unsqueeze(1), output2.unsqueeze(1)], dim=1)
        # 点积
        dot_products = torch.bmm(outputs, outputs.transpose(0, 2))
        # 计算正对角线和负对角线
        diag = torch.diagonal(dot_products, offset=0, dim1=1, dim2=2)
        # 计算对比损失的分母
        denominator = torch.sum(torch.exp(dot_products / self.temperature), dim=2)
        # 计算对比损失的分子
        numerator = torch.exp(diag / self.temperature)
        # 计算对比损失
        loss_contrastive = -torch.mean(torch.log(numerator / denominator))
        return loss_contrastive