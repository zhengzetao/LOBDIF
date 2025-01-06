from torch.utils.data import DataLoader
from sklearn import model_selection
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
import pdb
from torch.autograd import Variable

class MaskBatch():
    "object for holding a batch of data with mask during training"

    def __init__(self, src, pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt, pad, device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask

def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1, size, size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa

def normalize(data):
    """
    对列表数据的delta_time进行标准化或逆标准化。
    
    :param data: 输入的列表数据，格式为 [[time, type, delta_time], ...]
    :return: 标准化后的数据
    """
    data = np.array(data)  # 转换为 NumPy 数组以便处理
    last_column = data[:, :, -1]  # 提取最后一列
    # 标准化
    log_data = np.log(last_column + 1)  # 对数变化
    min_val = np.min(log_data)
    max_val = np.max(log_data)
    # pdb.set_trace()
    # 归一化
    # normalized_values = (log_data - min_val) / (max_val - min_val)
    last_column = (log_data - min_val) / (max_val - min_val)

    # 将处理后的最后一列放回原数据结构
    data[:, :, -1] = last_column
    data[:, :, -2] = data[:, :, -2].astype(np.int64)
    return data.tolist(), max_val, min_val  # 转换回列表格式

def inverse_normalize(data, MAX_MIN, device):
    data = np.array(data.detach().cpu())  # 转换为 NumPy 数组以便处理
    # last_column = data[:, :, 0]  # 提取最后一列
    # 逆标准化
    # pdb.set_trace()
    max_val = MAX_MIN[0]
    min_val = MAX_MIN[1]
    # 逆归一化
    log_data = min_val + (data * (max_val - min_val))
    # 反对数变换
    original_values = np.exp(log_data) - 1
    # last_column = np.exp(last_column) - 1
    # 将处理后的最后一列放回原数据结构
    # data[:, :, 0] = last_column
    return torch.tensor(original_values, device=device)
    # return data.tolist().to(device)  # 转换回列表格式

def df_to_list(df,len_of_record):
    records = []
    df = np.array(df)
    num_batches = len(df)//len_of_record
    for i in range(len(df) - len_of_record + 1):
        records.append(df[i:i + len_of_record, [0,1,-1]]) # only the date, event_type and delta_time are taken as input
    return(records)

def data_loader(dataset, len_of_record):
    ## training data loading ##
    df_train = pd.read_csv('./LOBSTER/WRDS/{}_train.csv'.format(dataset))
    # df_ = pd.read_csv('./LOBSTER/WRDS/{}_valid.csv'.format(dataset)) #only for MSFT
    # df_train = pd.concat([df_train, df_], ignore_index=True)  # ignore_index=True 重新索引
    df_train = df_train.tail(40000)
    delta_time = np.array(df_train.TIME_M[1:]) - np.array(df_train.TIME_M[:-1])
    delta_time[delta_time < 0] = 0
    df_train = df_train[1:]
    df_train['delta_time'] = delta_time #increase a new column
    train_record_list = df_to_list(df_train,len_of_record)
    train_record_list, _, _ = normalize(train_record_list)
    ## validation data loading ##
    df_valid = pd.read_csv('./LOBSTER/WRDS/{}_valid.csv'.format(dataset))
    df_valid = df_valid.head(2000)
    delta_time = np.array(df_valid.TIME_M[1:]) - np.array(df_valid.TIME_M[:-1])
    delta_time[delta_time < 0] = 0
    df_valid = df_valid[1:]
    df_valid['delta_time'] = delta_time #increase a new column
    val_record_list = df_to_list(df_valid,len_of_record)
    val_record_list, val_max, val_min = normalize(val_record_list)
    ## testing data loading ##
    df_test = pd.read_csv('./LOBSTER/WRDS/{}_test.csv'.format(dataset))
    df_test = df_test.head(2000)
    delta_time = np.array(df_test.TIME_M[1:]) - np.array(df_test.TIME_M[:-1])
    delta_time[delta_time < 0] = 0
    df_test = df_test[1:]
    df_test['delta_time'] = delta_time #increase a new column
    test_record_list = df_to_list(df_test,len_of_record)
    test_record_list, test_max, test_min = normalize(test_record_list)
    print("Number of rows in df_train:", len(df_train))
    print("Number of rows in df_valid:", len(df_valid))
    print("Number of rows in df_test:", len(df_test))

    return train_record_list, val_record_list, test_record_list, (val_max, val_min, test_max, test_min)

def parse_datasets(device,batch_size,dataset,train_percentage=0.8):
    total_dataset = dataset

    # Shuffle and split
    if train_percentage > 0:
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size= train_percentage, shuffle = False)
    else:
        test_data = total_dataset

    if train_percentage > 0:
        train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
            collate_fn = lambda batch: variable_time_collate_fn(batch, device))
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    else:
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    if train_percentage > 0:
        data_objects = {"dataset_obj": total_dataset,
                        "train_dataloader": inf_generator(train_dataloader),
                        "test_dataloader": inf_generator(test_dataloader),
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
    else:
        data_objects = {"dataset_obj": total_dataset,
                        "test_dataloader": inf_generator(test_dataloader),
                        "n_test_batches": len(test_dataloader)}
    return data_objects

def parse_datasets_separate(device,batch_size,train_dataset,val_dataset,test_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True,
        collate_fn = lambda batch: variable_time_collate_fn(batch, device))
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, device))

    data_objects = {"train_dataloader": inf_generator(train_dataloader),
                    "val_dataloader": inf_generator(val_dataloader),
                    "test_dataloader": inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_val_batches": len(val_dataloader),
                    "n_test_batches": len(test_dataloader)}

    return data_objects

def Batch2toModel(batch, transformer, type_num, device):

    # if opt.dim ==1:
    #     event_time_origin, event_time, lng = map(lambda x: x.to(device), batch)
    #     event_loc = lng.unsqueeze(dim=2)

    # if opt.dim==2:
    #     event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)

    #     event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)

    # if opt.dim==3:
    #     event_time_origin, event_time, lng, lat, height = map(lambda x: x.to(device), batch)

    #     event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2), height.unsqueeze(dim=2)),dim=-1)
    # event_time = batch['time_step']
    seq_event = batch['event']
    # seq_state = batch['mkt_state']
    batch_size = seq_event.shape[0]
    seq_event = seq_event.reshape(batch_size, -1, 1)
    event_time_origin = batch['delta_time']
    seq_event = seq_event.to(device)
    # event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    enc_out, mask = transformer(seq_event, event_time_origin) #enc_out=[?, ?, ?], mask=[?, ?, ?], seq_event=[512,50],event_time_origin=[512,50]
    # pdb.set_trace()
    enc_out_non_mask  = []
    event_time_non_mask = []
    event_non_mask = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        if length>1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length-1]] #[5060,192]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time_origin[index][1:length]] #[5060,1]
            event_non_mask += [i.unsqueeze(dim=0) for i in seq_event[index][1:length]] #[5060,2]
    enc_out_non_mask = torch.cat(enc_out_non_mask,dim=0) #[5060,192]
    event_time_non_mask = torch.cat(event_time_non_mask,dim=0) #[5060]
    event_non_mask = torch.cat(event_non_mask,dim=0) #[5060,2]
    event_time_non_mask = event_time_non_mask.reshape(-1,1,1) #[5060,1,1]
    event_non_mask = event_non_mask.reshape(-1,1,1) #[5060,1,2] args.dim=1
    event_type_one_hot = torch.nn.functional.one_hot(event_non_mask.squeeze(1).to(torch.int64),num_classes=type_num).type(torch.float)
    
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0],1,-1) #([5060, 1, 192]

    return event_time_non_mask, event_type_one_hot, enc_out_non_mask

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def variable_time_collate_fn(batch, device=torch.device("cuda")):
    D = 4
    T = 50
    batch = np.array(batch)

    data_dict = {
        # "time_step": torch.Tensor(batch[:,:,0] / 10).to(device),
        "event": torch.Tensor(batch[:,:,1] - 1).to(device),
        # "mkt_state": torch.Tensor(batch[:,:,-2]).to(device),
        "delta_time": torch.Tensor(batch[:,:,-1]).to(device)}

    return data_dict

def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()
    return data_dict

def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

########## for evaluation #############
def metrics_calculate(sampled_seq, event_time_non_mask, event_loc_non_mask):
    '''
    sampled_seq: (batch_size * seq_len, 1, 2)
    event_time_non_mask: (batch_size * seq_len, 1, 1)
    event_loc_non_mask: (batch_size * seq_len, 1, 1)
    '''
    # loss_test_all += loss.item() * event_time_non_mask.shape[0]
    real = (event_time_non_mask[:,0,:].detach().cpu() )
    gen = (sampled_seq[:,0,:1].detach().cpu() )
    # real = (event_time_non_mask[:,0,:].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
    # gen = (sampled_seq[:,0,:1].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
    assert real.shape==gen.shape
    # assert real.shape == a.shape
    # mae_temporal = torch.abs(real-gen).sum().item()
    
    # mae_temporal = torch.abs(real-gen).mean()
    rmse_temporal = ((real-gen)**2).sum().item()
    # log_gen = torch.log10(real)
    mae_temporal_mean = torch.abs(torch.log10(real+ 1e-10)- torch.log10(gen + 1e-10)).mean().item()
    # mae_temporal_mean = torch.abs(real-gen).mean().item()
    rmse_temporal_mean = rmse_temporal/sampled_seq.shape[0]
    # rmse_temporal_mean += ((real-sampled_seq_temporal_all)**2).sum().item()
    ####### calculate the predict accuracy -#############
    real = event_loc_non_mask[:,0,:].detach().cpu()
    # assert real.shape[1:] == torch.tensor(MIN[2:]).shape
    # real = (real + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
    gen = sampled_seq[:,0,1:].detach().cpu() # sampled_seq如何保证生成的是0-1之间
    # gen = (gen + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
    assert real.shape[0]==gen.shape[0]
    # assert real.shape==np.array(sampled_seq_spatial_all).shape
    # mae_spatial = torch.sqrt(torch.sum((real-gen)**2,dim=-1)).sum().item()
    # mae_spatial_mean = mae_spatial/sampled_seq.shape[0]
    ####### calculate the cross-entropy loss-#############
    cross_entropy_loss = torch.nn.CrossEntropyLoss()(gen.softmax(dim=1),real)
    # next_target_event = seq_event[:,-1,] #这里有错误，没有seq_event
    next_pred_event = torch.argmax(sampled_seq[:,0,1:],dim=1).detach().cpu()
    pred_accuracy = (torch.sum(torch.argmax(real,dim=1)==next_pred_event)).item()/len(next_pred_event)

    return cross_entropy_loss, pred_accuracy, mae_temporal_mean