import numpy
import torch
from utils import load_data, set_params, evaluate
from module import HeCo
import warnings
import datetime
import pickle as pkl
import os
import random

# import nni
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.cuda.empty_cache()

warnings.filterwarnings('ignore')
args = set_params()
# tuner_params = nni.get_next_parameter()
# args.alpha = tuner_params['alpha']
# args.beta = tuner_params['beta']

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



def train():
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    feats_dim_list = [i.shape[1] for i in feats]
    P = int(len(mps))
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    print("alpha: ", args.alpha)
    print("beta: ", args.beta)
    with open('log.txt', 'a') as file:
        file.write("dataset: " + str(args.dataset) + '\n')
        file.write("alpha: "+ str(args.alpha) + '\n')
        file.write("beta: " + str(args.beta) + '\n')
    # print("len(feats): ", len(feats)) #3
    # print("feats[0].shape: ", feats[0].shape)#torch.Size([4019, 1902])
    # print("feats[1].shape: ", feats[1].shape)#torch.Size([7167, 7167])
    # print("feats[2].shape: ", feats[2].shape)#torch.Size([60, 60])
    
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                    P, args.sample_rate, args.nei_num, args.tau, args.lam, args.alpha, args.beta)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = [feat.cuda() for feat in feats]
        mps = [mp.cuda() for mp in mps]
        pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()
        loss = model(feats, pos, mps, nei_index)
        # nni.report_intermediate_result(loss)
        print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    # loss = model(feats, pos, mps, nei_index)
    # nni.report_final_result(loss)
    model.load_state_dict(torch.load('HeCo_'+own_str+'.pkl'))
    model.eval()
    os.remove('HeCo_'+own_str+'.pkl')
    embeds = model.get_embeds(feats, mps)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    with open('log.txt', 'a') as file:
        file.write("Total time: " + str(time) + "s" + '\n')
    print("Total time: ", time, "s")
    
    if args.save_emb:
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()


if __name__ == '__main__':
    train()
