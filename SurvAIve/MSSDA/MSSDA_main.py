"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     <MSSDA_main.py>
  Authors:  <Ammar Shaker (Ammar.Shaker@neclab.eu)>

NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

"""
#python MSSDA/MSSDA_main.py --RNA_Type 0 --data_path data/TCGA/miRNA --model SurvAIve.MSSDA.SurviveSDI.SurviveSDI --lr 0.0001 --lr_alphas 0.001 --batch_size 80 --feat_extract_size 200 --MaxSizePerDomain -1 --nb_experiments 5 --largeVal 1 --epochs_adapt 20 --current_experiment 0 --cuda True --Test_domain_id 0 --percent_target 0.0 --target_domain_type 1 --target_domain_weight 1.0 --save_path results/MSSDAmiRNA --norm_alphas 2
#python MSSDA/MSSDA_main.py --RNA_Type 1 --data_path data/TCGA/mRNA/mRNA_mtl.mat --model SurvAIve.MSSDA.SurviveSDI.SurviveSDI --lr 0.0001 --lr_alphas 0.001 --batch_size 80 --feat_extract_size 200 --MaxSizePerDomain -1 --nb_experiments 5 --largeVal 1 --epochs_adapt 20 --current_experiment 0 --cuda True --Test_domain_id 0 --percent_target 0.0 --target_domain_type 1 --target_domain_weight 1.0 --save_path results/MSSDAmRNA --norm_alphas 2
#--RNA_Type 0 --data_path C:/Data/TCGA/miRNAdataset/Raw_21_types --model MSSDA.SurviveSDI.SurviveSDI --lr 0.0001 --lr_alphas 0.0001 --batch_size 80 --feat_extract_size 200 --MaxSizePerDomain -1 --nb_experiments 5 --largeVal 1 --epochs_adapt 30 --current_experiment 0 --cuda False --Test_domain_id 0 --percent_target 0.0 --target_domain_type 1 --target_domain_weight 1.0 --save_path results --norm_alphas 2
import sys
sys.path.append('.')
sys.path.append('../')
import numpy as np
import scipy.io
from os.path import isfile, join
from os import listdir
import os
import math
import torch
import datetime
import argparse
import random
import logging
import pickle
from pathlib import Path
from itertools import product
from typing import List, Tuple
from pydoc import locate
from SurvAIve.MSSDA.SurviveSDI import get_default_predictor, get_default_feature_extractor, get_defualt_discriminator
from SurvAIve.MSSDA.utils.enums import Target_domain_type
from SurvAIve.MSSDA.utils.utils import split_source_target_survival, batch_loader_survival_supervised_weighted, read_miRNA_survival_file, str2bool, get_pkl_dic, c_index

def get_percentage_target_domain(X_t: np.ndarray, y_t: np.ndarray, event_t: np.ndarray,
                                 percentage_t: float, min_event_perc: float = 0.2) -> tuple:
    """
    Split the target domain dataset into two parts based on a specified percentage and minimum event percentage.

    The function randomly selects a subset of the target domain dataset (`X_t`, `y_t`, `event_t`) and splits it
    into two parts. The first split retains a percentage of the data specified by `percentage_t` and has a minimum
    event percentage specified by `min_event_perc`.

    Args:
        X_t (np.ndarray): Features (input data) of the target domain.
        y_t (np.ndarray): Target (label) data of the target domain.
        event_t (np.ndarray): Event indicators, where 1 indicates an event occurred and 0 indicates censored data.
        percentage_t (float): The desired percentage (between 0 and 1) to retain in the first split.
        min_event_perc (float, optional): The minimum event percentage required for the first split.
            Defaults to 0.2 (20%).

    Returns:
        tuple: A tuple containing five elements:
            - Xt (np.ndarray): Features of the first split.
            - yt (np.ndarray): Target data of the first split.
            - eventt (np.ndarray): Event indicators of the first split.
            - indices_included (np.ndarray): The indices of the samples included in the first split.
            - indices_remaining (np.ndarray): The indices of the samples not included in the first split (remaining samples).
    """
    size= math.ceil(X_t.shape[0]*percentage_t)
    while True:
        r_order = np.arange(X_t.shape[0])
        np.random.shuffle(r_order)

        Xt, yt, eventt = X_t[r_order[:size], :], y_t[r_order[:size]], event_t[r_order[:size]]
        if  eventt.sum()/size>min_event_perc:
            return Xt, yt, eventt,r_order[:size],r_order[size:]

def check_file_str(folder: str, ID: str) -> bool:
    """
    Check if there are any log files in the specified folder that contain the given ID in their filenames.

    Args:
        folder (str): The folder path to check for log files.
        ID (str): The ID to look for in the filenames of the log files.

    Returns:
        bool: True if at least one log file with the given ID is found in the folder, False otherwise.
    """
    if not os.path.exists(folder):
        return False
    onlylogs = [f for f in listdir(folder) if isfile(join(folder, f)) and ".log" in f and ID in f]
    return len(onlylogs)>0


def get_miRNA(args: argparse.Namespace) -> Tuple[List, List, List, List]:
    """
    The function reads CSV files from the specified data path, extracts relevant
    data columns, and returns them as lists. The function assumes that the CSV
    files contain data related to miRNA_old, survival time, and event/censoring status.
    The domain_list contains identifiers associated with each data point.

    Args:
        args (argparse.Namespace): Command-line arguments passed to the function.

    Returns:
        Tuple[List, List, List, List]: A tuple containing the following lists:
            - X_survival: List of features from miRNA_old data.
            - y_survival: List of survival time data.
            - event_survival: List of event/censoring status data.
            - domain_list: List of domains or identifiers associated with the data.
    """
    onlyfiles = [f for f in listdir(args.data_path) if isfile(join(args.data_path, f)) and ".csv" in f]
    onlyfiles.sort()
    X_survival, y_survival, event_survival, domain_list = [], [], [], []
    for f in onlyfiles:
        filename= f
        data,df = read_miRNA_survival_file(os.path.join(args.data_path, f))
        X_survival += [data[:,2:]]
        y_survival += [data[:,0]]
        event_survival += [data[:,1]]
        domain_list += [f.replace(".csv","")]
        print(f)

    if args.MaxSizePerDomain>0 :
        for i in range(len(X_survival)):
            print(X_survival[i].shape,y_survival[i].shape)
            if X_survival[i].shape[0] > args.MaxSizePerDomain:
                X_survival[i] = X_survival[i][:args.MaxSizePerDomain]
                y_survival[i] = y_survival[i][:args.MaxSizePerDomain]
                event_survival[i] = event_survival[i][:args.MaxSizePerDomain]
            print(X_survival[i].shape,y_survival[i].shape)
    return X_survival, y_survival, event_survival, domain_list

def get_mRNA(args: argparse.Namespace) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Read mRNA data from a MATLAB .mat file and return relevant data.
    The function loads data from a MATLAB .mat file specified by the 'data_path' argument.
    It extracts relevant data columns and returns them as lists of numpy arrays.
    The 'domain_list' contains names or identifiers associated with each domain.

    Args:
        args (argparse.Namespace): Command-line arguments passed to the function.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]: A tuple containing the following lists:
            - X_survival: List of mRNA features as numpy arrays.
            - y_survival: List of survival time data as numpy arrays.
            - event_survival: List of event/censoring status data as numpy arrays.
            - domain_list: List of domain names associated with the data.
   """
    mat = scipy.io.loadmat(args.data_path)

    #Reading the training data files
    X_survival, y_survival, event_survival, domain_list = [], [], [], []
    data = mat["X"]
    domain_list = ["KIRC", "OV", "GBM", "LUAD", "LUSC", "BRCA", "HNSC", "LAML"]
    n_domains = len(data)
    for i in range(n_domains):
        domain = data[i][0]
        #X_survival += [np.expand_dims(domain[:,3:].astype(np.int8), axis=1)]
        X_survival += [domain[:,3:].astype(np.int8)]
        y_survival += [domain[:,0].astype(np.int16)]
        event_survival += [domain[:,1].astype(np.int8)]

    if args.MaxSizePerDomain>0:
        for i in range(len(X_survival)):
            print(X_survival[i].shape,y_survival[i].shape)
            if X_survival[i].shape[0] > args.MaxSizePerDomain:
                X_survival[i] = X_survival[i][:args.MaxSizePerDomain]
                y_survival[i] = y_survival[i][:args.MaxSizePerDomain]
                event_survival[i] = event_survival[i][:args.MaxSizePerDomain]
            print(X_survival[i].shape,y_survival[i].shape)

    return X_survival, y_survival, event_survival, domain_list

def initialize(args: argparse.Namespace, ID: str) -> Tuple[logging.Logger, logging.FileHandler, dict, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], torch.device, str]:
    """
    Initialize the experiment by setting up logging, random seeds, data loading, and other parameters.

    Args:
        args (argparse.Namespace): Command-line arguments for configuration.
        ID (str): Identifier for the experiment.

    Returns:
        Tuple[logging.Logger, logging.FileHandler, dict, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], torch.device, str]: A tuple containing:
            - logger: A logger for logging experiment information.
            - handler: File handler for logger.
            - dict_args: A dictionary containing model parameters.
            - X_survival: List of input features.
            - y_survival: List of survival time data.
            - event_survival: List of event/censoring status data.
            - domain_list: List of domain names or identifiers.
            - device: Torch device for computation (CPU or GPU).
            - date: Current date and time for experiment identification.

    This function initializes an experiment, including setting up logging, random seeds,
    loading data based on the specified RNA_Type, and configuring model parameters.
    """
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"_"+str(int(np.random.uniform(low=0.0, high=10000.0, size=1)[0]))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.RNA_Type==0:
        save_path_log = os.path.join(args.save_path, 'miRNA_' + args.model + '_' + date + "_" +ID +'.log')
    else:
        save_path_log = os.path.join(args.save_path, 'mRNA_' + args.model + '_' + date + "_" +ID +'.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # create a file handler
    handler = logging.FileHandler(os.path.join(save_path_log))
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info(args)

    # initialize seeds
    torch.manual_seed(args.current_experiment)
    print(args.current_experiment)
    np.random.seed(args.current_experiment)
    random.seed(args.current_experiment)
    if args.cuda:
        torch.cuda.manual_seed_all(args.current_experiment)

    if args.cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    if args.RNA_Type==0:
        X_survival, y_survival, event_survival, domain_list = get_miRNA(args)
    else:
        X_survival, y_survival, event_survival, domain_list = get_mRNA(args)

    n_domains = len(domain_list)

    dict_args= {'input_dim': X_survival[0].shape[1], 'output_dim': 1, 'n_sources': n_domains-1,
         'min_y_value': -np.inf, 'max_y_value': np.inf, "norm_alphas":args.norm_alphas}

    return logger, handler, dict_args, X_survival, y_survival, event_survival, domain_list, device, date

def run(args, ID):
    logger, handler, dict_args, X_survival, y_survival, event_survival, domain_list, device, date = initialize(args, ID)
    dicAll_res = {}

    #Number of epochs
    target_domain_type = Target_domain_type(args.target_domain_type)

    i = args.Test_domain_id
    domain = domain_list[i]
    logger.info('\n ----------------------------- %i / %i -----------------------------'%(args.current_experiment+1, args.nb_experiments))
    #Split source and target
    X_s, X_t, y_s, y_t, event_s, event_t = split_source_target_survival(X_survival, y_survival, event_survival, i, device, merge=False)

    #Initialize model
    dict_args.update({'feature_extractor': get_default_feature_extractor(args.feat_extract_size, args.bottleneck_size, input_dim=X_survival[0].shape[1], Dropout_per=args.Dropout_per),
            'prediction_head': get_default_predictor(args.bottleneck_size, output_dim=1),
            'discriminator_head': get_defualt_discriminator(args.bottleneck_size, output_dim=1),
            'logger': logger})
    print(device)
    print(args.model)
    Model = locate(args.model)
    model = Model(dict_args).to(device)
    model.init(args, device)

    logger.info('---- '+ domain+ ' ----')

    #Alternated training
    logger.info('------------Alternated training------------')

    if target_domain_type==Target_domain_type.no_target_learning:
        test_index, rem_index = np.zeros((0)),np.arange(X_t.shape[0])
    else:
        xlab_t, ylab_t, eventlab_t, test_index, rem_index = get_percentage_target_domain(X_t, y_t, event_t,args.percent_target)

    xlab_t, ylab_t, eventlab_t, = X_t[test_index,:], y_t[test_index], event_t[test_index]
    x_rem_t, y_rem_t, event_rem_t = X_t[rem_index,:], y_t[rem_index], event_t[rem_index]

    # Computing the initial risk and c-index on source and target domains
    risk_pred_s = [model.predict(X_s[s]) for s in range(len(X_s))]
    c_index_s = [c_index(-risk_pred_s[s],y_s[s],event_s[s]) for s in range(len(X_s))]
    risk_pred_t = model.predict(X_t)
    c_index_t = c_index(-risk_pred_t,y_t,event_t)

    # Computing the initial risk and c-index on the target domains only on the remaining samples, x_rem_t,
    # that are not used for trining as a percentage
    risk_pred_t_rem = model.predict(x_rem_t)
    c_index_t_rem = c_index(-risk_pred_t_rem,y_rem_t, event_rem_t)

    dicAll_res[domain_list[i]+"_Before_Learning"] = [c_index_s,c_index_t,c_index_t_rem]

    for epoch in range(args.epochs_adapt):
        logger.info(epoch)
        model.train()
        loader = batch_loader_survival_supervised_weighted(X_s, y_s, event_s, xlab_t, ylab_t, eventlab_t, args.target_domain_weight, device,  batch_size = args.batch_size, shuffle = True, random_state=epoch)
        model.run_epoch(loader, X_t, event_t)

        if (epoch+1)%1==0:
            model.eval()
            source_loss, disc2_SDI = model.compute_all_losses(X_s, X_t, y_s, event_s, event_t, args.largeVal, device)

            risk_pred_s = [model.predict(X_s[s]) for s in range(len(X_s))]
            c_index_s = [c_index(-risk_pred_s[s],y_s[s],
                                 event_s[s]) for s in range(len(X_s))]

            risk_pred_t = model.predict(X_t)
            c_index_t = c_index(-risk_pred_t,y_t,event_t)

            risk_pred_t_rem = model.predict(x_rem_t)
            c_index_t_rem = c_index(-risk_pred_t_rem,y_rem_t, event_rem_t)
            dicAll_res[domain_list[i]+"_"+str(args.current_experiment)+"_"+str(epoch)] = [[source_loss, disc2_SDI],
                                                                                          model.source_weights.cpu().detach().numpy().copy(), c_index_s, c_index_t, c_index_t_rem]
            logger.info('Epoch: %i/ (h_pred); source_loss: %.3f ; disc2_SDI: %.3f'
                      %(epoch+1, source_loss.item(), disc2_SDI.item()))
            logger.info("c_index_s: "+str(c_index_s))
            logger.info("c_index_t: "+str(c_index_t))

    model.eval()
    source_loss, disc2_SDI = model.compute_all_losses(X_s, X_t, y_s, event_s, event_t, args.largeVal, device)

    risk_pred_s = [model.predict(X_s[s]) for s in range(len(X_s))]
    c_index_s = [c_index(-risk_pred_s[s],y_s[s],event_s[s]) for s in range(len(X_s))]

    risk_pred_t = model.predict(X_t)
    c_index_t = c_index(-risk_pred_t,y_t,event_t)

    risk_pred_t_rem = model.predict(x_rem_t)
    c_index_t_rem = c_index(-risk_pred_t_rem,y_rem_t, event_rem_t)

    dicAll_res[domain+"_Final"] = [[source_loss, disc2_SDI],
                                   model.source_weights.cpu().detach().numpy().copy(), c_index_s, c_index_t, c_index_t_rem]

    dicAll_res["args"] = args
    dicAll_res["sorted_domains"] = domain_list
    # Its important to use binary mode
    if args.RNA_Type==0:
        save_path_result = os.path.join(args.save_path, 'miRNA_' +args.model + '_' + date + "_" +ID+ '.pkl')
    else:
        save_path_result = os.path.join(args.save_path, 'mRNA_' + args.model + '_' + date + "_" +ID+ '.pkl')

    dbfile = open(save_path_result, 'wb')

    # source, destination
    pickle.dump(dicAll_res, dbfile)
    dbfile.close()
    logger.removeHandler(handler)

def build_args(element,args):
    current_experiment,Test_domain_id,target_domain_type,\
                                percent_target, model = element
    args.current_experiment, args.Test_domain_id  = int(current_experiment), int(Test_domain_id)
    args.target_domain_type = int(target_domain_type)
    args.percent_target = float(percent_target)
    args.model = model

    return args

def get_paerser_arguments():
    parser = argparse.ArgumentParser(description='Multi-source Survival Analysis transfer learning')

    # model details
    parser.add_argument('--model', type=str, default='SurvAIve.MSSDA.SurviveSDI.SurviveSDI', help='model to train')
    parser.add_argument('--Test_domain_id', type=int, default=0, help='Test domain id')
    # experiment details
    parser.add_argument('--nb_experiments', type=int, default=5, help='Number experiments')

    parser.add_argument('--current_experiment', type=int, default=0, help='Current experiment, also the seed')

    # optimizer parameters influencing all models
    parser.add_argument('--batch_size', type=int, default=80, help='the amount of items received by the algorithm at one time')

    parser.add_argument('--epochs_adapt', type=int, default=20, help='epochs_adapt')
    parser.add_argument('--bottleneck_size', type=int, default=20, help='bottleneck_size')
    parser.add_argument('--Dropout_per', type=float, default=0.1, help='Dropout percentage')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_alphas', type=float, default=0.001, help='learning rate')
    parser.add_argument('--norm_alphas', type=int, default=2, help='Norm used for alphas, 1 or 2')
    parser.add_argument('--feat_extract_size', type=int, default=200, help='the number of units in the feature extractor')
    parser.add_argument('--target_domain_type', type=int, default=1, help='no_target_learning = 1; target_only_in_loss = 2; target_only_in_divergence = 3; target_in_both = 4')
    parser.add_argument('--target_domain_weight', type=float, default=1.0, help='how much weight should be put for the target samples')
    parser.add_argument('--percent_target', type=float, default=0.0, help='percentage of the supervised data from the target task')
    parser.add_argument('--largeVal', type=int, default=1, help='large Value')

    # experiment parameters
    parser.add_argument('--cuda',  type=str2bool, default=False,help='Use GPU?')
    parser.add_argument('--log_every', type=int, default=100, help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/', help='save models at the end of training')

    parser.add_argument('--data_path', type=str, default='data/', help='path of the miRNA_old data files, or the Matlab mRNA file')
    parser.add_argument('--RNA_Type',  type=str2bool, default=0,help='0:miRNA_old, 1:mRNA')

    # data parameters
    parser.add_argument('--MaxSizePerDomain', type=int, default=-1, help='the maximum size for each domain. -1 means this attribute is ignored.')

    # Parse arguments while ignoring the '-f' argument added by Jupyter
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    args = get_paerser_arguments()

    if int(args.norm_alphas) not in [1,2]:
        raise argparse.ArgumentTypeError('Int value expected either 1 or 2.')

    if os.path.exists(args.save_path):
        already = get_pkl_dic(args.save_path)
    else:
        already = {}
    key = str(args.model) +"_"+ str(args.Test_domain_id) +"_"+str(args.target_domain_type) \
            +"_"+str(args.target_domain_weight)+"_"+str(args.percent_target) \
            +"_"+str(args.lr)+"_"+str(args.lr_alphas) \
            +"_"+str(args.norm_alphas)+ "_"+str(args.current_experiment)
    #run(args, key)
    if args.target_domain_type==1 and float(args.percent_target)!=float("0.00"):
        raise argparse.ArgumentTypeError('The setting target_domain_type=1 does not accept the supervised learning (percent_target!=0).')
    if args.target_domain_type!=1 and float(args.percent_target)==float("0.00"):
        raise argparse.ArgumentTypeError("The setting target_domain_type!=1 does not accept the fully unsupervised learning (percent_target=0).")

    key = str(args.model) +"_"+ str(args.Test_domain_id) +"_"+str(args.target_domain_type) \
        +"_"+str(args.target_domain_weight)+"_"+str(args.percent_target) \
        +"_"+str(args.lr)+"_"+str(args.lr_alphas) \
        +"_"+str(args.norm_alphas)+ "_"+str(args.current_experiment)
    print(key)
    if key not in already.keys():# and not check_file_str(args.save_path,key):
        print("Start",key)
        print(args.current_experiment, args.Test_domain_id, args.target_domain_type, args.percent_target, args.model)
        run(args, key)

#    args = get_paerser_arguments()
#    exps = range(args.nb_experiments)
#    Test_domain_ids = range(8)
#    #target_domain_types = [1,2,3,4]
#    target_domain_types = [1,4]
#    percent_targets = ["0.00","0.05","0.1","0.15","0.2","0.25"]
#    model_names = ["SurvAIve.MSSDA.SurviveSDI.SurviveSDI"]
#
#    args.batch_size, args.feat_extract_size, args.nb_experiments = 80, 200, 5
#    args.largeVal, args.epochs_adapt, args.cuda, args.norm_alphas = 1, 20, True, 2
#    args.target_domain_weight = 1.0
#    args.lr, args.lr_alphas =  0.0001, 0.001
#
#
#    args.RNA_Type = 0
#    args.data_path = r"/home/ashaker/ProtoSurvive/data/TCGA/miRNA_old"
#    args.save_path = r"/home/ashaker/ProtoSurvive/results/MSSDAmiRNA"
#
#
#    args.RNA_Type = 1
#    args.data_path = r"/home/ashaker/ProtoSurvive/data/TCGA/mRNA/mRNA_mtl.mat"
#    args.save_path = r"/home/ashaker/ProtoSurvive/results/MSSDAmRNA"
#    args.largeVal, args.epochs_adapt, args.cuda, args.norm_alphas = 1, 20, True, 2
#    args.lr, args.lr_alphas =  0.0001, 0.001
#
#    if int(args.norm_alphas) not in [1,2]:
#        raise argparse.ArgumentTypeError('Int value expected either 1 or 2.')
#
#
#    #pdb.set_trace()
#    if os.path.exists(args.save_path):
#        already = get_pkl_dic(args.save_path)
#    else:
#        already = {}
#
#    key = str(args.model) +"_"+ str(args.Test_domain_id) +"_"+str(args.target_domain_type) \
#            +"_"+str(args.target_domain_weight)+"_"+str(args.percent_target) \
#            +"_"+str(args.lr)+"_"+str(args.lr_alphas) \
#            +"_"+str(args.norm_alphas)+ "_"+str(args.current_experiment)
#    #run(args, key)
#
#
#
#    Current_index=0
#    for i, element in enumerate(product(exps,Test_domain_ids,target_domain_types,
#                                    percent_targets,model_names)):
#        current_experiment,Test_domain_id,target_domain_type, percent_target, model = element
#        args.current_experiment, args.Test_domain_id  = int(current_experiment), int(Test_domain_id)
#        args.target_domain_type = int(target_domain_type)
#        args.percent_target = float(percent_target)
#        args.model = model
#
#        if args.target_domain_type==1 and float(args.percent_target)!=float("0.00"):
#            continue
#        if args.target_domain_type!=1 and float(args.percent_target)==float("0.00"):
#            continue
#
#        key = str(args.model) +"_"+ str(args.Test_domain_id) +"_"+str(args.target_domain_type) \
#            +"_"+str(args.target_domain_weight)+"_"+str(args.percent_target) \
#            +"_"+str(args.lr)+"_"+str(args.lr_alphas) \
#            +"_"+str(args.norm_alphas)+ "_"+str(args.current_experiment)
#        print(key)
#        if key not in already.keys() and not check_file_str(args.save_path,key):
#            print("Start",key)
#            print(args.current_experiment, args.Test_domain_id, args.target_domain_type, args.percent_target, args.model)
#            run(args, key)
#            print(Current_index)
#
#        Current_index+=1
