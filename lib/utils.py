import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims), dtype=np.float32)
    y = np.zeros(shape = (num_sample, Q, nodes, dims), dtype=np.float32)
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def read_meta(path):
    df = pd.read_csv(path)
    if 'lat' in df.columns and 'lng' in df.columns:
        locations = df[['lat', 'lng']].values
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        locations = df[['latitude', 'longitude']].values
    else:
        locations = df.iloc[:, 1:3].values
    return locations.astype(np.float32)

def construct_adj(data, num_node):
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix-tem_matrix.mean())/tem_matrix.std())
    return tem_matrix

def augmentAlign(dist_matrix, auglen):
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def reorderData(parts_idx, mxlen, adj, sps):
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps-part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps-part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0])+sps*i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx

def frequency_patching(train_data, num_patches, meta_file_path=None):
    sample_steps = min(train_data.shape[0], 3000)
    data = train_data[:sample_steps, :, 0].transpose(1, 0)
    fft_features = np.abs(np.fft.rfft(data, axis=1))[:, :32]
    mean_feat = np.mean(data, axis=1, keepdims=True)
    std_feat = np.std(data, axis=1, keepdims=True)
    
    locations = read_meta(meta_file_path)
    
    fft_norm = StandardScaler().fit_transform(fft_features)
    stat_norm = StandardScaler().fit_transform(np.concatenate([mean_feat, std_feat], axis=1))
    coord_norm = StandardScaler().fit_transform(locations)
    
    combined_features = np.concatenate([fft_norm * 1.0, stat_norm * 3.0, coord_norm * 5.0], axis=1)
    
    kmeans = KMeans(n_clusters=num_patches, n_init=20, random_state=2024)
    labels = kmeans.fit_predict(combined_features)
    
    parts = []
    cluster_counts = [np.sum(labels == i) for i in range(num_patches)]
    max_len = max(cluster_counts)
    if max_len % 2 != 0: max_len += 1
    
    for i in range(num_patches):
        indices = np.where(labels == i)[0]
        if len(indices) == 0: indices = np.array([0])
        indices = np.sort(indices)
        parts.append(indices)
            
    return parts, max_len

def loadData(filepath, metapath, P, Q, train_ratio, test_ratio, adjpath, recurtimes, tod, dow, sps, log):
    locations = read_meta(metapath)
    
    if os.path.exists(adjpath):
        adj = np.load(adjpath, allow_pickle=True)
    else:
        num_node = locations.shape[0]
        adj = np.eye(num_node) 

    Traffic = np.load(filepath, allow_pickle=True)['data'][..., :1]
    num_step = Traffic.shape[0]
    
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    
    trainData = Traffic[: train_steps]
    valData = Traffic[train_steps : train_steps + val_steps]
    testData = Traffic[-test_steps :]
    
    TE = np.zeros([num_step, Traffic.shape[1], 2]) 
    for i in range(num_step):
        TE[i, :, 0] = i % tod           
        TE[i, :, 1] = (i // tod) % dow  
        
    trainTE = TE[: train_steps]
    valTE = TE[train_steps : train_steps + val_steps]
    testTE = TE[-test_steps :]

    num_nodes = trainData.shape[1]
    target_patch_num = int(num_nodes / sps)
    if target_patch_num < 1: target_patch_num = 1

    parts_idx, mxlen = frequency_patching(trainData, target_patch_num, metapath)
    real_patch_num = len(parts_idx)
    ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(parts_idx, mxlen, adj, mxlen)
    
    partition_map = np.zeros(num_nodes, dtype=int)
    for community_id, node_indices in enumerate(parts_idx):
        partition_map[node_indices] = community_id

    trainX, trainY = seq2instance(trainData, P, Q)
    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testX, testY = seq2instance(testData, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)

    mean, std = np.mean(trainX), np.std(trainX)

    return trainX, trainY, trainXTE, trainYTE, \
           valX, valY, valXTE, valYTE, \
           testX, testY, testXTE, testYTE, \
           mean, std, \
           ori_parts_idx, reo_parts_idx, reo_all_idx, \
           real_patch_num, mxlen, \
           locations, partition_map 