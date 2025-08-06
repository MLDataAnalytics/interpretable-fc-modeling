import sys
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.io as sio

from region_interpret_model import region_int
from model_loss import mse_loss, huber_loss, ncc_loss
from fc_dataset import fcDataset



class Config:
    pass


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':
    #
    config = Config()
    #
    data_id = sys.argv[1]            #"fold0" ...
    
    config.y_idx = int(sys.argv[2])  #3 #2 #1
    config.y_type = "pc" + str(config.y_idx)

    config.dat_dir = sys.argv[3]     #r"/cbica/home/lihon/comp_space_recover/ABCD_comp/fc_comp_qc"
    config.dat_file = sys.argv[4]    #r"/cbica/home/lihon/comp_space_recover/ABCD_comp/dl_prediction/data_cog_abcc_bl_cv/abcd_fc_tra_" + data_id + ".csv"

    config.fc_mode = sys.argv[5]     # cortex, all, or subcortical
    config.out_base_dir = sys.argv[6]

    config.val_file = sys.argv[7]

    #
    config.batch_size = 32
    config.lr = 0.0005
    config.n_epochs = 200

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

    out_suffix = data_id + "_" + config.loss_type + "_" + config.y_type + "_" + config.optim_type
    out_suffix = out_suffix + f"_regAtt_{config.reg_att}"
    config.output_dir = config.out_base_dir + r"/model_" + out_suffix

    config.start_weights_file = None
    config.checkpoint_interval = 2
    #

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # target data
    fc_ds = fcDataset(config.dat_file, config.dat_dir, 'fc_p2p', config.y_idx)
    trainloader = torch.utils.data.DataLoader(fc_ds,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=1)
    len_dataset = len(trainloader.dataset)

    # validation data
    val_ds = fcDataset(config.val_file, config.dat_dir, 'fc_p2p', config.y_idx)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=1)
    len_val = len(val_loader.dataset)
    epoch_idx = []
    val_mae = []
    val_corr = []
    cor_best = 1e-6
    mae_best = 1e6

    # model
    model = region_int(d_r=config.d_r, d_in=config.d_in, d_out=config.d_out)

    if config.start_weights_file is not None:
        model.load_state_dict(torch.load(config.start_weights_file))
        print('Continue training, start with: ', config.start_weights_file)

    model = model.to(device)

    if config.optim_type == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), config.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), config.lr)

    if config.loss_type == "mse":
        pred_loss = mse_loss()
    else:
        pred_loss = huber_loss(1.0)
    pred_loss = pred_loss.to(device)

    start_time = time.time()
    for epoch in range(1, config.n_epochs+1):
        running_loss = 0.0
        # training
        model.train()

        epoch_start_time = time.time()
        for i, i_batch in enumerate(trainloader, 0):
            batch_dat = i_batch[0]      # B x N_p x N_p
            batch_y = i_batch[1]        # B  
            
            n_b, n_p, n_fn = batch_dat.shape
            if not n_b==config.batch_size:
                continue

            X_tgt = batch_dat.float().to(device)
            if config.fc_mode in ["cortex", "all"]:
                X_tgt = X_tgt[:,0:config.d_r,:][:,:,0:config.d_in]
            else:
                X_tgt = X_tgt[:,-config.d_r:,:][:,:,0:config.d_in]

            y_tgt = batch_y.float().to(device).reshape(-1, 1)
            y_tgt = y_tgt * torch.ones(1, config.d_r).to(device)

            optimizer.zero_grad()
            
            r_id = F.one_hot(torch.arange(0, config.d_r), config.d_r).to(dtype=torch.float32).to(device)   # R x R
            y_pred_m, y_pred, r_att = model(X_tgt, r_id)

            loss_p = pred_loss(y_pred, y_tgt)
            loss_p_m = pred_loss(y_pred_m, y_tgt[:,[0]])
            loss = loss_p + loss_p_m * config.reg_att

            loss_ncc = ncc_loss(y_pred_m, y_tgt[:,[0]])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 5 == 0:
                print(f'[epoch {epoch}, iter {i+1:5d}] running_loss: {running_loss / (i+1):.3f}, \
                         loss: {loss.item():.3f}, pred_m_loss: {loss_p_m.item():.3f}, pred_loss: {loss_p.item():.3f}, ncc_loss: {loss_ncc.item():.3f}')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch} took {epoch_duration:.2f} seconds.")

        if epoch % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), config.output_dir
                       + '/weights_{}.pth'.format(epoch))

            # on evaluation
            model.eval()

            with torch.no_grad():
                y_pred_mat = np.zeros((len_val, config.d_r))
                r_att_mat = np.zeros((len_val, config.d_r))
                y_pred_m_mat = np.zeros((len_val,1))

                y_true_mat = np.zeros((len_val,1))

                for vi, vi_batch in enumerate(val_loader, 0):
                    batch_dat = vi_batch[0]      # B x N_p x N_p
                    batch_y = vi_batch[1]        # B  
                    
                    n_b, n_p, n_fn = batch_dat.shape

                    X_tgt = batch_dat.float().to(device)
                    if config.fc_mode in ["cortex", "all"]:
                        X_tgt = X_tgt[:,0:config.d_r,:][:,:,0:config.d_in]
                    else:
                        X_tgt = X_tgt[:,-config.d_r:,:][:,:,0:config.d_in]

                    y_tgt = batch_y.float().to(device).reshape(-1, 1)
                    y_tgt = y_tgt * torch.ones(1, config.d_r).to(device)
                    
                    r_id = F.one_hot(torch.arange(0, config.d_r), config.d_r).to(dtype=torch.float32).to(device)   # R x R
                    y_pred_m, y_pred, r_att = model(X_tgt, r_id)

                    y_pred_np = y_pred.cpu().detach().numpy()
                    y_tgt_np = y_tgt.cpu().detach().numpy()

                    vi_st = vi * config.batch_size
                    vi_ed = vi_st + n_b
                    y_pred_mat[vi_st:vi_ed] = y_pred_np
                    y_true_mat[vi_st:vi_ed] = y_tgt_np[:,[0]]

                    y_pred_m_np = y_pred_m.cpu().detach().numpy()
                    y_pred_m_mat[vi_st:vi_ed] = y_pred_m_np
                    r_att_np = r_att.cpu().detach().numpy()
                    r_att_mat[vi_st:vi_ed] = r_att_np

                # save validation prediction
                sio.savemat(config.output_dir + "/validation_prediction_{}.mat".format(epoch),
                            {"y_pred":y_pred_mat, "y_true":y_true_mat,
                             "y_pred_m":y_pred_m_mat, "r_att":r_att_mat})
                y_pred_m = y_pred_m_mat

                # validation mae and corr
                mae = np.mean(np.abs(y_pred_m-y_true_mat))
                corr = np.corrcoef(np.reshape(y_pred_m,(-1,)), np.reshape(y_true_mat,(-1,)))[0,1]
                print(f'region-int -- MAE: {mae:.3f}, corr: {corr:.3f}')

                epoch_idx.append(epoch)
                val_mae.append(mae)
                val_corr.append(corr)

                #
                cor_ratio = corr / cor_best
                mae_ratio = mae_best / mae
                if (cor_ratio+mae_ratio)/2 > 1.0:
                    cor_best = corr
                    mae_best = mae
                    torch.save(model.state_dict(), config.output_dir + '/weights_val_opt.pth')
    #
    val_mae = np.array(val_mae)
    val_corr = np.array(val_corr)
    sio.savemat(config.output_dir + "/validation_performance.mat",
                {"epoch_idx":epoch_idx, "val_mae":val_mae,
                 "val_corr":val_corr})

    print("Model training finished.")

    end_time = time.time()
    total_training_duration = end_time - start_time
    print(f"Total training time: {total_training_duration:.2f} seconds.")
