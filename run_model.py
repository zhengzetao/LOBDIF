import torch
import pandas as pd
import numpy as np
import utils
import os
from torch.optim import AdamW, Adam
from model import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import time
import pdb
import argparse
from utils import *
from model.model_CT4LSTM_PPP import CT4LSTM_PPP

def define_args():
    parser = argparse.ArgumentParser('LOB event prediction')
    parser.add_argument('--dataset',  type=str, default="INTC", help="dataset used")
    parser.add_argument('--class_loss_weight', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--model', type=str, default='LOBDIF')
    parser.add_argument('--mkt_state', action='store_true', default=True)
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--dim', type=int, default=1, help='', choices = [1,2,3])
    parser.add_argument('--timesteps', type=int, default=100, help='')
    parser.add_argument('--samplingsteps', type=int, default=50, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--loss_type', type=str, default='l2',choices=['l1','l2','Euclid'], help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--niter', type=int, default=100, help='')
    parser.add_argument('--d_model', type=int, default=64, help='')
    parser.add_argument('--d_rnn', type=int, default=256, help='')
    parser.add_argument('--nhead', type=int, default=4, help='')
    parser.add_argument('--nkv', type=int, default=16, help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine',choices=['linear','cosine'], help='')
    return parser.parse_args()

# def main(dataset,class_loss_weight,model,mkt_state,seed,gpu):
def main(dataset="MSFT",class_loss_weight=1,model='LOBDIF',mkt_state=True,seed=0,gpu=0):
    args = define_args()
    # args.dataset = dataset
    # args.class_loss_weight = class_loss_weight
    args.model = model
    args.mkt_state = mkt_state
    args.seed = seed
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
    # device = torch.device("cuda:{}".format(args.cuda_id))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda')
    len_of_record = 50
    # batch_size = 32#512
    batch_size = args.batch_size
    niter = args.niter#15#200
    lr = args.lr
    model_path_known_t = './models/{}_{}_mkt-{}_coef-{}_{}rms{}_new.mdl'.format(args.dataset,args.model,args.mkt_state,args.class_loss_weight,args.lr,2e-3)
    log_path = './logs/{}_{}_mkt-{}_coef-{}_{}rms{}_new.log'.format(args.dataset,args.model,args.mkt_state,args.class_loss_weight,args.lr,2e-3)
    
    # Ensure the directory exists
    log_dir = os.path.dirname(log_path)
    model_dir = os.path.dirname(model_path_known_t)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = open(log_path, "w")

    # train_record_list = []
    # val_record_list = []
    # test_record_list = []
    # for i in range(5): # 5 is the batch of dataset,0-2 for training, 3 for validation and 4 for testing
    #     df_ = pd.read_csv('./LOBSTER/parsed_event_df/{}_event_df_{}.csv'.format(args.dataset,i+1),index_col=0)
    #     delta_time = np.array(df_.time[1:]) - np.array(df_.time[:-1])
    #     df = df_[1:]
    #     df['delta_time'] = delta_time #increase a new column
    #     # pdb.set_trace()
    #     # df.time = delta_time
    #     if i < 3:
    #         train_record_list = df_to_list(df,len_of_record) + train_record_list
    #         train_record_list = normalize(train_record_list)
    #     elif i == 3:
    #         val_record_list = df_to_list(df,len_of_record)
    #         val_record_list = normalize(val_record_list)
    #     elif i == 4:
    #         test_record_list = df_to_list(df,len_of_record)
    #         test_record_list = normalize(test_record_list)
    train_record_list, val_record_list, test_record_list, MAX_MIN = data_loader(args.dataset, len_of_record)
    type_num = int(np.max(np.array(train_record_list)[:,:,1].astype(np.int64)))
    print("the number of tpye is", type_num)
    data_obj = parse_datasets_separate(device,batch_size,train_record_list[:(len(train_record_list)//batch_size)*batch_size],val_record_list[:(len(val_record_list)//batch_size)*batch_size],test_record_list[:(len(test_record_list)//batch_size)*batch_size])

    st_diffusion= ST_Diffusion(n_steps=args.timesteps, dim=1+type_num, condition = True, cond_dim=args.d_model).to(device)
    diffusion = GaussianDiffusion_ST(st_diffusion, loss_type = args.loss_type, seq_length = 1+type_num, timesteps = args.timesteps,
        sampling_timesteps = args.samplingsteps, objective = args.objective, beta_schedule = args.beta_schedule).to(device)
    transformer = Transformer_ST(d_model=args.d_model, d_rnn=args.d_rnn, d_inner=128, n_layers=4, n_head=args.nhead, d_k=args.nkv, d_v=args.nkv, dropout=0.1,
        device=device, loc_dim = type_num, CosSin = True).to(device)


    models = Model_all(transformer,diffusion)
    models = models.cuda()
    warmup_steps = 5
    # training
    optimizer = AdamW(models.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_valid = 1e20

    # model = CT4LSTM_PPP(num_events = 4, hidden_dim = 16, input_embed_dim = 16)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    average_loss_train = []
    min_loss_valid = 1e20
    # val_accuracy_list_known_t = []
    for itr in range(1, data_obj['n_train_batches'] * (niter + 1)):
    # for itr in tqdm(range(1, data_obj['n_train_batches'] * (niter + 1)), desc = 'Training'):
        models.train()
        optimizer.zero_grad()
        batch_dict = get_next_batch(data_obj["train_dataloader"])
        update_learning_rate(optimizer, decay_rate=0.999, lowest=2e-3)
        # intens, cells, cell_targets, outputs, decays = model(batch_dict, args.mkt_state, device = torch.device("cuda"))
        # loss, hawkes_loss, pred_accuracy = model.compute_loss(batch_dict, intens, cells, cell_targets, outputs, decays, args.class_loss_weight)
        event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch_dict, models.transformer, type_num, device)
        # sampled_seq = models.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
        loss = models.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)
        # ce_loss, accuracy_score, temporal_loss = metrics_calculate(sampled_seq, event_time_non_mask, event_loc_non_mask)
        loss.backward()
        optimizer.step()
        average_loss_train.append(loss.item())
        # cross_entropy_loss_train.append(ce_loss)
        # temporal_loss_train.append(temporal_loss)
        # accuracy_train.append(accuracy_score)
        # if itr % data_obj['n_train_batches'] == 0:
        if itr % data_obj['n_train_batches'] == 0:
            torch.cuda.empty_cache()
            train_message = 'Epoch {:04d} diffusion_loss: {:.4f}'.format(itr // data_obj['n_train_batches'],sum(average_loss_train) / len(average_loss_train))
            average_loss_train = []
            # train_message = 'Epoch {:04d} diffusion_loss: {:.4f} Accuracy_score: {:.4f} cross entropy loss: {:.4f} temporal loss: {:.4f}'.format(itr // data_obj['n_train_batches'],sum(average_loss_train) / len(average_loss_train),sum(accuracy_train) / len(accuracy_train),sum(cross_entropy_loss_train) / len(cross_entropy_loss_train),sum(temporal_loss_train) / len(temporal_loss_train))
            print(train_message)
            logger.write(train_message+'\n')
            if itr % (10 * data_obj['n_train_batches']) == 0:
                # target_array = []
                # pred_array_known_time = []
                cross_entropy_loss_valid, accuracy_valid, temporal_loss_valid = [], [], []
                valid_loss_all = 0.0
                for i in range(data_obj['n_val_batches']):
                    models.eval()
                    batch_dict = get_next_batch(data_obj['val_dataloader'])
                    # target, pred_known_time, hawkes_loss_val= model.read_predict(batch_dict, args.mkt_state, metrics=False)
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch_dict, models.transformer, type_num, device)
                    sampled_seq = models.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    event_time_inverse_normalize = inverse_normalize(event_time_non_mask, MAX_MIN[:2], device)
                    sampled_seq[:,0,0] = inverse_normalize(sampled_seq[:,0,0], MAX_MIN[:2], device)
                    ce_loss, accuracy_score, temporal_loss = metrics_calculate(sampled_seq, event_time_inverse_normalize, event_loc_non_mask)
                    loss = models.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)
                    # target_array.append(target)
                    cross_entropy_loss_valid.append(ce_loss)
                    accuracy_valid.append(accuracy_score)
                    temporal_loss_valid.append(temporal_loss)
                    valid_loss_all += loss.item() 
                    # pred_array_known_time.append(pred_known_time)
                    # hawkes_loss_val_list.append(hawkes_loss_val*len(batch_dict['time_step']))
                # target_array = np.hstack(target_array).reshape(-1)
                # pred_array_known_time = np.hstack(pred_array_known_time).reshape(-1)
                valid_message = 'Validation Cross_entropy loss {:.4f}, Accuracy_score {:.4f}, Temporal loss {:.4f}'.format((sum(cross_entropy_loss_valid) / len(cross_entropy_loss_valid)), (sum(accuracy_valid) / len(accuracy_valid)), (sum(temporal_loss_valid) / len(temporal_loss_valid)))
                print(valid_message)
                logger.write(valid_message+'\n')
                # val_accuracy_list_known_t.append(sum(hawkes_loss_val_list) / len(val_record_list))
                #early stop
                if valid_loss_all > min_loss_valid:
                    early_stop += 1
                    if early_stop >= 100:
                        break    
                else:
                    early_stop = 0
                    torch.save(models,model_path_known_t)
                # if val_accuracy_list_known_t[-1] == np.min(val_accuracy_list_known_t):
                #     torch.save(models,model_path_known_t)
                min_loss_valid = min(min_loss_valid, valid_loss_all)

            # accuracy_list = []
            # average_loss = []

    # models = torch.load(model_path_known_t) #临时屏蔽
    # target_array = []
    # pred_array_known_time = []
    # pred_array_unknown_time = []
    # hawkes_loss_test_list = []
    # error_dt_test_list = []
                cross_entropy_loss_test, accuracy_test, temporal_loss_test, sampled_time_consume = [], [], [], []
                start_time = time.time()
                for i in range(data_obj['n_test_batches']):
                    models.eval()
                    batch_dict = get_next_batch(data_obj['test_dataloader'])
                    # target, pred_known_time, pred_unknown_time, hawkes_loss_test, error_dt_test = model.read_predict(batch_dict, args.mkt_state,metrics=True)
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch_dict, models.transformer, type_num, device)
                    sample_start_time = time.time()
                    sampled_seq = models.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    event_time_inverse_normalize = inverse_normalize(event_time_non_mask, MAX_MIN[2:], device)
                    sampled_seq[:,0,0] = inverse_normalize(sampled_seq[:,0,0], MAX_MIN[2:], device)
                    sampled_time_consume.append(time.time() - sample_start_time)
                    # pdb.set_trace()
                    # print(sampled_seq[:3,0,1:],event_loc_non_mask[:3,0,:])
                    ce_loss, accuracy_score, temporal_loss = metrics_calculate(sampled_seq, event_time_inverse_normalize, event_loc_non_mask)
                    cross_entropy_loss_test.append(ce_loss)
                    accuracy_test.append(accuracy_score)
                    temporal_loss_test.append(temporal_loss)
                elapsed_time = time.time() - start_time
                print('average sampling time:',  sum(sampled_time_consume) / len(sampled_time_consume))
                test_message = 'Test Cross_entropy_loss, {:.4f} Accuracy_score {:.4f}, Temporal error {:.4f}, Test time consume {:.4f} '.format((sum(cross_entropy_loss_test) / len(cross_entropy_loss_test)), (sum(accuracy_test) / len(accuracy_test)),  (sum(temporal_loss_test) / len(temporal_loss_test)), elapsed_time)
                logger.write(test_message+'\n')
                print(test_message)
    logger.close()
if __name__ == '__main__':
    main()








