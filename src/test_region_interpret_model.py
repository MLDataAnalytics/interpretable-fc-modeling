import sys
import os

import torch
#import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io as sio

from region_interpret_model import region_int
from fc_dataset import fcDataset



class Config:
    pass


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    config = Config()
    #
    data_id = sys.argv[1]                   #"fold0", 1, 2, 3, 4

    config.y_idx = int(sys.argv[2])         #3 #2 #1
    config.y_type = "pc" + str(config.y_idx)

    config.dat_dir = sys.argv[3]            #r"/cbica/home/lihon/comp_space_recover/ABCD_comp/fc_comp_qc"
    config.dat_file = sys.argv[4]

    config.fc_mode = sys.argv[5]            # all, cortex, or subcortical
    config.out_base_dir = sys.argv[6]

    config.batch_size = 32

    config.reg_att = 1.0

    if config.fc_mode == "all":
        config.d_in = 352
    else:
        config.d_in = 333

    if config.fc_mode == "all":
        config.d_r = 352
    elif config.fc_mode == "cortex":
        config.d_r = 333
    else:
        config.d_r = 19

    config.d_out = 1
    
    config.loss_type = "mse"
    config.optim_type = "Adam"

    out_suffix = data_id + '_' + config.loss_type + "_" + config.y_type + "_" + config.optim_type
    out_suffix = out_suffix + f"_regAtt_{config.reg_att}"

    config.weights_file = config.out_base_dir + r"/model_" + out_suffix + "/weights_val_opt.pth"
    #config.weights_file = config.out_base_dir + r"/model_" + out_suffix + "/weights_200.pth"
    config.output_dir = config.out_base_dir + r"/results_tes_" + out_suffix

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # testing data
    fc_ds = fcDataset(config.dat_file, config.dat_dir, 'fc_p2p', config.y_idx)
    trainloader = torch.utils.data.DataLoader(fc_ds,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=1)
    len_dataset = len(trainloader.dataset)
    print(r"Number of testing subjects: " + str(len_dataset))

    # model
    model = region_int(d_r=config.d_r, d_in=config.d_in, d_out=config.d_out)

    model.load_state_dict(torch.load(config.weights_file))
    model = model.to(device)
    model.eval()

    #
    y_pred_mat = np.zeros((len_dataset, config.d_r))
    r_att_mat = np.zeros((len_dataset, config.d_r))
    y_pred_m_mat = np.zeros((len_dataset,1))

    y_true_mat = np.zeros((len_dataset,1))

    for i, i_batch in enumerate(trainloader, 0):
        batch_dat = i_batch[0]      # B x N_p x N_p
        batch_y = i_batch[1]        # B  
        
        n_b, n_p, n_fn = batch_dat.shape

        X_tgt = batch_dat.float().to(device)
        if config.fc_mode in ["all", "cortex"]:
            X_tgt = X_tgt[:,0:config.d_r,:][:,:,0:config.d_in]
        else:
            X_tgt = X_tgt[:,-config.d_r:,:][:,:,0:config.d_in]

        y_tgt = batch_y.float().to(device).reshape(-1, 1)
        y_tgt = y_tgt * torch.ones(1, config.d_r).to(device)
        
        r_id = F.one_hot(torch.arange(0, config.d_r), config.d_r).to(dtype=torch.float32).to(device)   # R x R
        y_pred_m, y_pred, r_att = model(X_tgt, r_id)

        y_pred_np = y_pred.cpu().detach().numpy()
        y_tgt_np = y_tgt.cpu().detach().numpy()

        i_st = i * config.batch_size
        i_ed = i_st + n_b
        y_pred_mat[i_st:i_ed] = y_pred_np
        y_true_mat[i_st:i_ed] = y_tgt_np[:,[0]]

        y_pred_m_np = y_pred_m.cpu().detach().numpy()
        y_pred_m_mat[i_st:i_ed] = y_pred_m_np
        r_att_np = r_att.cpu().detach().numpy()
        r_att_mat[i_st:i_ed] = r_att_np

    sio.savemat(config.output_dir + "/testing_prediction.mat",
                {"y_pred":y_pred_mat, "y_true":y_true_mat,
                 "y_pred_m":y_pred_m_mat, "r_att":r_att_mat})

    # mae and corr
    mae = np.mean(np.abs(y_pred_m_mat-y_true_mat))
    corr = np.corrcoef(np.reshape(y_pred_m_mat,(-1,)), np.reshape(y_true_mat,(-1,)))[0,1]
    print(f'region-int -- MAE: {mae:.3f}, corr: {corr:.3f}')
    print("Finished.")
