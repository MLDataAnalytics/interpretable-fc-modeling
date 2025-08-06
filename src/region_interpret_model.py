import torch
import torch.nn as nn
import torch.nn.functional as F



class UnitNet(nn.Module):
    def __init__(self, d_in=333, d_out=1):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.l1 = nn.Linear(d_in, d_out)

    def forward(self, x):
        out = self.l1(self.dropout(x))
        return out


class region_int(nn.Module):
    def __init__(self, d_r=333, d_in=333, d_out=1):    
        super(region_int, self).__init__()
        self.d_r = d_r
        self.fc_region_out = nn.ModuleList([UnitNet(d_in, d_out) for _ in range(d_r)])
        self.fc_att = nn.Linear(d_r, 1, bias=False)
        nn.init.constant_(self.fc_att.weight, 0)

    def forward(self, x, r_id=None):
        r_out = []
        for i in range(self.d_r):
            i_o = self.fc_region_out[i](x[:,i,:])       # B x out
            i_o = torch.unsqueeze(i_o, 1)               # B x 1 x out
            r_out.append(i_o)
        r_out = torch.cat(r_out, 1)                     # B x R x out

        # fusing
        #initial version
        #r_att = torch.sigmoid(self.fc_att.weight)               # 1 x R
        #r_att = r_att / (torch.sum(r_att,1,True) + 1e-8)        # 1 x R
        # v2
        r_att = self.fc_att(r_id)                               # R x 1
        r_att = torch.reshape(r_att, (1,-1))                    # 1 x R
        r_att = torch.sigmoid(r_att)
        r_att = r_att / (torch.sum(r_att,1,True) + 1e-8)

        out = r_out * torch.unsqueeze(r_att, 2)         # B x R x out
        out = torch.sum(out, 1, False)                  # B x out

        r_out = torch.squeeze(r_out, -1)

        return out, r_out, r_att
