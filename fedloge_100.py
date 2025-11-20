# python version 3.7.1
# -*- coding: utf-8 -*-

import os
import copy
import numpy as np
import random
import torch

import torch.nn as nn
from tqdm import tqdm
from options import args_parser, args_parser_cifar10
from util.update_baseline import LocalUpdate, globaltest, localtest
from util.fedavg import *
from util.dataset import *
from model.build_model import build_model
from util.dispatch import *
from util.losses import *
from util.etf_methods import ETF_Classifier
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

load_switch = False  # True / False
save_switch = True # True / False
cls_switch = "SSE-C"
pretrain_cls = False
dataset_switch = 'cifar100' # cifar10 / cifar100
aggregation_switch = 'fedavg' # fedavg / class_wise
global_test_head = 'g_head'  # g_aux / g_head
internal_frozen = False  # True / False
loss_switch = "None" # focous_loss / any others

def get_acc_file_path(args):

    rootpath = './temp/'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
 
    if args.balanced_global:
        rootpath+='global_' 
    rootpath += 'fl'
    if args.beta > 0:
        rootpath += "_LP_%.2f" % (args.beta)
    fpath =  rootpath + '_acc_{}_{}_cons_frac{}_iid{}_iter{}_ep{}_lr{}_N{}_{}_seed{}_p{}_dirichlet{}_IF{}_Loss{}.txt'.format(
        args.dataset, args.model, args.frac, args.iid, args.rounds, args.local_ep, args.lr, args.num_users, args.num_classes, args.seed, args.non_iid_prob_class, args.alpha_dirichlet, args.IF, args.loss_type)
    return fpath

def plot_metrics(overall, many, medium, few, args, names=('overall', 'many', 'medium', 'few')):
    overall, many, medium, few = np.array(overall), np.array(many), np.array(medium), np.array(few)
    """
    Plot four metric curves and display maximum value information
    
    Parameters:
    overall (np.array): overall metric array
    many (np.array): many metric array
    medium (np.array): medium metric array
    few (np.array): few metric array
    names (tuple): names of the four metrics, default ('overall', 'many', 'medium', 'few')
    """
    # Validate that input arrays have the same length
    if not (len(overall) == len(many) == len(medium) == len(few)):
        raise ValueError("All input arrays must have the same length")
    if len(args.load) > 3:
        n = 1
    else:
        n = args.relabel_start
    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        n = min(n, len(overall))  # Ensure it does not exceed array length

    rounds = np.arange(0, len(overall))
    
    plt.figure(figsize=(12, 6))
    
    # Plot curves
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    h1, = plt.plot(rounds, overall, color=colors[0], label=names[0])
    h2, = plt.plot(rounds, many, color=colors[1], label=names[1])
    h3, = plt.plot(rounds, medium, color=colors[2], label=names[2])
    h4, = plt.plot(rounds, few, color=colors[3], label=names[3])

    # Calculate global maximum
    global_max_idx = np.argmax(overall)
    global_max_round = global_max_idx
    global_max_value = overall[global_max_idx]
    global_other = (many[global_max_idx], medium[global_max_idx], few[global_max_idx])

    # Calculate maximum in first n rounds
    n_max_info = ""
    if n is not None:
        local_max_idx = np.argmax(overall[:n])
        local_max_round = local_max_idx
        local_max_value = overall[local_max_idx]
        local_other = (many[local_max_idx], medium[local_max_idx], few[local_max_idx])
        
        # Add marker line for first n rounds
        plt.axvline(x=local_max_round, color='gray', linestyle='--', alpha=0.5)
        plt.scatter(local_max_round, local_max_value, 
                   color=colors[0], zorder=5, s=50,
                   edgecolor='black', linewidth=0.5)

        # Construct information for first n rounds
        n_max_info = (
            f"\n\n max clean before round {n}:\n"
            f"{names[0]} peak: Round {local_max_round} ({local_max_value:.3f})\n"
            f"At Round {local_max_round}:\n"
            f"{names[1]}: {local_other[0]:.3f}\n"
            f"{names[2]}: {local_other[1]:.3f}\n"
            f"{names[3]}: {local_other[2]:.3f}"
        )

    # Global maximum marker
    plt.axvline(x=global_max_round, color='black', linestyle=':', alpha=0.8)
    plt.scatter(global_max_round, global_max_value,
               color=colors[0], zorder=5, s=50,
               edgecolor='black', linewidth=0.5)

    # Legend information
    global_info = (
        f"model_id:{args.id}\n"
        f"all max overall:\n"
        f"{names[0]} peak: Round {global_max_round} ({global_max_value:.3f})\n"
        f"At Round {global_max_round}:\n"
        f"{names[1]}: {global_other[0]:.3f}\n"
        f"{names[2]}: {global_other[1]:.3f}\n"
        f"{names[3]}: {global_other[2]:.3f}"
        f"{n_max_info}"
    )

    # Configure figure
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Combine legend
    from matplotlib.lines import Line2D
    legend_handles = [
        h1, h2, h3, h4,
        Line2D([], [], color='black', linestyle=':', label='Global Peak Marker'),
        Line2D([], [], color='gray', linestyle='--', label=f'First {n} Rounds Peak' if n else '')
    ]
    
    plt.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=9
    )
    
    # Add right side text
    plt.text(
        x=1.05, 
        y=0.5,
        s=global_info,
        transform=plt.gca().transAxes,
        verticalalignment='center',
        fontsize=9
    )
    
    plt.tight_layout()
    plt.savefig(args.id+'.pdf',format='pdf')


if __name__ == '__main__':
    # parse args
    if dataset_switch == 'cifar100':
        args = args_parser()
    elif dataset_switch == 'cifar10':
        args = args_parser_cifar10()

    overallacc,manyacc,mediumacc,fewacc = [],[],[],[]
    global_test_head = args.ghead
    internal_frozen = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    fpath = get_acc_file_path(args)
    print(fpath)

    # myDataset containing details and configs about dataset
    datasetObj = myDataset(args)
    if args.balanced_global:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_balanced_dataset(datasetObj.get_args())
    else:
        dataset_train, dataset_test, dict_users, dict_localtest = datasetObj.get_imbalanced_dataset(datasetObj.get_args())
         
    print(len(dict_users))

    # build model
    model = build_model(args) 
    # Freeze specific layers
    if internal_frozen:
        model.layer1[0].conv1.weight.requires_grad = False

        # If Conv2d layer has bias, also freeze it
        if model.layer1[0].conv1.bias is not None:
            model.layer1[0].conv1.bias.requires_grad = False
    
    # copy weights
    w_glob = model.state_dict()
    w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]

    # Training
    m = max(int(args.frac * args.num_users), 1) 
    prob = [1/args.num_users for j in range(args.num_users)]

    in_features = model.linear.in_features
    out_features = model.linear.out_features

    if cls_switch == "SSE-C":
        # Initialize ETF classifier
        etf = ETF_Classifier(in_features, out_features) 
        # Create linear layer with ETF classifier's ori_M as weight
        g_head = nn.Linear(in_features, out_features).to(args.device) 
        sparse_etf_mat = etf.gen_sparse_ETF(feat_in = in_features, num_classes = out_features, beta=0.6)

        g_head.weight.data = sparse_etf_mat.to(args.device)
        g_head.weight.data = g_head.weight.data.t()
    else:
        print("STOP")
        exit()

    if pretrain_cls == True:
        g_head.load_state_dict({k.replace('linear.', ''): v for k, v in torch.load("/home/zikaixiao/zikai/aapfl/pfed_lastest/demo.pth").items() if 'linear' in k})


    g_aux = nn.Linear(in_features, out_features).to(args.device)
    
    l_heads = []
    for i in range(args.num_users):
        l_heads.append(nn.Linear(in_features, out_features).to(args.device))
    load_switch = args.load
    print('pre_id:', load_switch)
    if len(load_switch) > 3:
        rndstart = args.relabel_start - 1
        load_dir = "./output/cifar10/"
        model = torch.load(load_dir + args.load +"model_" + "best" + ".pth").to(args.device)
        g_head = torch.load(load_dir + args.load+"g_head_" + "best" + ".pth").to(args.device)
        g_aux = torch.load(load_dir + args.load+"g_aux_" + "best" + ".pth").to(args.device)
        for i in range(args.num_users):
            l_heads[i] = torch.load(load_dir + args.load+"l_head_" + str(i) + "best.pth").to(args.device)
        w_glob = model.state_dict()
        w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]
    else:
        rndstart = 0
    max_overall = 0
    is_best = 0
    for rnd in tqdm(range(rndstart, args.rounds)):
        g_auxs = []
        w_locals = []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)

        # Local training       
        g_head.train()
        for client_id in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[client_id])
            w_local, g_aux_temp, l_heads[client_id], loss_local = local.update_weights_gaux(net=copy.deepcopy(model).to(args.device), 
                                                                                            g_head = copy.deepcopy(g_head).to(args.device), 
                                                                                            g_aux = copy.deepcopy(g_aux).to(args.device), 
                                                                                            l_head = l_heads[client_id], 
                                                                                            epoch=args.local_ep, args=args, loss_switch = loss_switch, 
                                                                                            isbest= is_best, net_id = client_id, rnd = rnd)
            g_auxs.append(g_aux_temp)
            w_locals.append(w_local)
        g_head.eval()
        # Aggregation 
        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        w_glob = FedAvg_noniid(w_locals, dict_len)

        if aggregation_switch == 'fedavg':
            g_aux = FedAvg_noniid_classifier(g_auxs, dict_len)
        elif aggregation_switch == 'class_wise':
            g_aux = cls_norm_agg(g_auxs, dict_len, l_heads=l_heads, distributions = datasetObj.training_set_distribution)

        # Assign
        w_locals = [copy.deepcopy(w_glob) for i in range(args.num_users)]

        # Global test
        model.load_state_dict(copy.deepcopy(w_glob))

        if global_test_head == 'g_head':
            acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_head).to(args.device), dataset_test, args, dataset_class = datasetObj)
        elif global_test_head == 'g_aux':
            acc_s2, global_3shot_acc = globaltest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_aux).to(args.device), dataset_test, args, dataset_class = datasetObj)

        overallacc.append(acc_s2)
        manyacc.append(global_3shot_acc["head"])
        mediumacc.append(global_3shot_acc["middle"])
        fewacc.append(global_3shot_acc["tail"])
        plot_metrics(overallacc, manyacc, mediumacc, fewacc, args)
        if max_overall <= acc_s2:
            is_best = 1
            max_overall = acc_s2
        else:
            is_best = 0
        # Local test 
        acc_list = []
        f1_macro_list = []
        f1_weighted_list = []
        acc_3shot_local_list = []
        for i in range(args.num_users):
            model.load_state_dict(copy.deepcopy(w_locals[i]))
            acc_local, f1_macro, f1_weighted, acc_3shot_local = localtest(copy.deepcopy(model).to(args.device), copy.deepcopy(g_aux).to(args.device), copy.deepcopy(l_heads[i]).to(args.device), dataset_test, dataset_class = datasetObj, idxs=dict_localtest[i], user_id = i)
            acc_list.append(acc_local)
            f1_macro_list.append(f1_macro)
            f1_weighted_list.append(f1_weighted)
            acc_3shot_local_list.append(acc_3shot_local)

        if save_switch == True and is_best:
            save_dir = "./output/cifar10/"
            if rnd >= args.relabel_start:
                torch.save(model, save_dir + args.id +"model_" + "bestN" + ".pth")
                torch.save(g_head, save_dir + args.id+"g_head_" + "bestN" + ".pth")
                torch.save(g_aux, save_dir + args.id+"g_aux_" + "bestN" + ".pth")
                for i in range(args.num_users):
                    torch.save(l_heads[i], save_dir + args.id+"l_head_" + str(i) + "bestN.pth")
            else:
                torch.save(model, save_dir + args.id +"model_" + "best" + ".pth")
                torch.save(g_head, save_dir + args.id+"g_head_" + "best" + ".pth")
                torch.save(g_aux, save_dir + args.id+"g_aux_" + "best" + ".pth")
                for i in range(args.num_users):
                    torch.save(l_heads[i], save_dir + args.id+"l_head_" + str(i) + "best.pth")

        if save_switch == True and rnd == 499:
            for i in range(args.num_users):
                torch.save(l_heads[i], save_dir + args.id+"l_head_" + str(i) + "499.pth")
        # Calculate acc_3shot_local
        avg3shot_acc={"head":0, "middle":0, "tail":0}
        divisor = {"head":0, "middle":0, "tail":0}
        for i in range(len(acc_3shot_local_list)):
            avg3shot_acc["head"] += acc_3shot_local_list[i]["head"][0]
            avg3shot_acc["middle"] += acc_3shot_local_list[i]["middle"][0]
            avg3shot_acc["tail"] += acc_3shot_local_list[i]["tail"][0]
            divisor["head"] += acc_3shot_local_list[i]["head"][1]
            divisor["middle"] += acc_3shot_local_list[i]["middle"][1]
            divisor["tail"] += acc_3shot_local_list[i]["tail"][1]
        avg3shot_acc["head"] /= divisor["head"]
        avg3shot_acc["middle"] /= divisor["middle"]
        avg3shot_acc["tail"] /= divisor["tail"]
    torch.cuda.empty_cache()
