import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
import torch.fft
from tqdm import tqdm

try:
    from model import FreqCoT
    from utils import log_string, loadData, _compute_loss, metric
except ImportError:
    from models.model import FreqCoT
    from lib.utils import log_string, loadData, _compute_loss, metric

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        log_string(log, '\n------------ Loading Data (AMP + Grad Accumulation) -------------')
        
        data_pack = loadData(
            self.traffic_file, self.meta_file, self.input_len, self.output_len,
            self.train_ratio, self.test_ratio, self.adj_file, self.recur_times,
            self.tod, self.dow, self.spa_patchsize, log
        )
        
        (
            self.trainX, self.trainY, self.trainXTE, self.trainYTE,
            self.valX, self.valY, self.valXTE, self.valYTE,
            self.testX, self.testY, self.testXTE, self.testYTE,
            self.mean, self.std,
            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx,
            self.real_patch_num, self.mxlen,
            self.locations, self.partition_map
        ) = data_pack


        loc_mean = np.mean(self.locations, axis=0, keepdims=True)
        loc_std = np.std(self.locations, axis=0, keepdims=True) + 1e-5
        self.locations = (self.locations - loc_mean) / loc_std
        print("DEBUG: Locations normalized for model stability.")

        self.spa_patchnum = self.real_patch_num
        self.spa_patchsize = self.mxlen
        
        log_string(log, f'Data Loaded. Patches: {self.spa_patchnum}x{self.spa_patchsize}')

        if str(self.cuda) == '-1' or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("WARNING: Running on CPU!")
        else:
            self.device = torch.device(f"cuda:{self.cuda}")
            print(f"SUCCESS: Running on GPU -> {torch.cuda.get_device_name(self.device)}")

        self.build_model()

    def build_model(self):
        print(f"DEBUG: Building FreqCoT Model...")
        model_configs = vars(self)
        self.model = FreqCoT(
            configs=model_configs,
            partition_map=self.partition_map,
            node_coords=self.locations
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # [新增] 混合精度 Scaler
        self.scaler = torch.cuda.amp.GradScaler()

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[40, 80], gamma=0.2
        )

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]

        print("DEBUG: Keeping data on CPU (FP32 Mode)...")
        self.trainX_tensor = torch.from_numpy((self.trainX - self.mean) / self.std).float()
        self.trainY_tensor = torch.from_numpy(self.trainY).float()
        self.trainXTE_tensor = torch.from_numpy(self.trainXTE).float()
        
        self.valX_tensor = torch.from_numpy((self.valX - self.mean) / self.std).float()
        self.valY_tensor = torch.from_numpy(self.valY).float()
        self.valXTE_tensor = torch.from_numpy(self.valXTE).float()
        print("DEBUG: Done.")

        target_batch_size = 64 
        real_batch_size = self.batch_size 
        accum_steps = max(1, target_batch_size // real_batch_size)
        
        print(f"DEBUG: Gradient Accumulation Enabled. Real BS: {real_batch_size} | Accum Steps: {accum_steps} | Equiv BS: {real_batch_size * accum_steps}")

        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            train_l_sum, batch_count = 0.0, 0
            start = time.time()
            
            permutation = np.random.permutation(num_train)
            num_batch = math.ceil(num_train / self.batch_size)
            pbar = tqdm(total=num_batch, desc=f"Epoch {epoch}/{self.max_epoch}", leave=False)
            
            self.optimizer.zero_grad() 
            
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_train, (batch_idx + 1) * self.batch_size)
                indices = permutation[start_idx : end_idx]

                NormX = self.trainX_tensor[indices].to(self.device, non_blocking=True)
                Y = self.trainY_tensor[indices].to(self.device, non_blocking=True)
                TE = self.trainXTE_tensor[indices].to(self.device, non_blocking=True)
                

                y_hat = self.model(NormX, TE)
                

                pred_y = y_hat * self.std + self.mean
                real_y = Y

                loss = _compute_loss(real_y, pred_y)
                

                loss = loss / accum_steps 
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batch:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_l_sum += loss.item() * accum_steps
                batch_count += 1
                
                pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})
                pbar.update(1)
            
            pbar.close()
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step()
            
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []
        num_batch = math.ceil(num_val / self.batch_size)
        
        with torch.no_grad():
            for batch_idx in range(num_batch):
                start_idx = batch_idx * self.batch_size
                end_idx = min(num_val, (batch_idx + 1) * self.batch_size)
                
                NormX = self.valX_tensor[start_idx : end_idx].to(self.device, non_blocking=True)
                TE = self.valXTE_tensor[start_idx : end_idx].to(self.device, non_blocking=True)
                

                with torch.cuda.amp.autocast():
                    y_hat = self.model(NormX, TE)
                
                pred.append(y_hat.cpu().numpy() * self.std + self.mean)
                label.append(self.valY_tensor[start_idx : end_idx].numpy())
        
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        
        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse, mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log, 'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args, _ = parser.parse_known_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)


    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['tps'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['tpn'])
    parser.add_argument('--factors', type=int, default = config['param']['factors'])
    parser.add_argument('--recur_times', type = int, default = config['param']['recur'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['sps'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['spn'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['id'])
    parser.add_argument('--node_dims', type=int, default = config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dd'])


    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])

    args = parser.parse_args()
    
    log = open(args.log_file, 'w')
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))
    solver.train()