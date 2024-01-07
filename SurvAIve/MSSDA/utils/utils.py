"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     <utils.py>
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
import numpy as np
import torch
import argparse
import pandas as pd
import pickle
from typing import List, Tuple, Union, Dict, Generator
from os.path import isfile, join
from os import listdir
from io import BytesIO
from lifelines.utils import concordance_index

def str2bool(v: Union[str, bool]) -> bool:
    """
    Convert a string representation of a boolean value to a boolean.

    Args:
        v (str or bool): The string to be converted or an already existing boolean value.

    Returns:
        bool: The corresponding boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input value is not a recognized boolean string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_miRNA_survival_file(filename: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Reads a CSV file and returns the data as both a NumPy array and a Pandas DataFrame.

    Args:
        filename (str): The path to the CSV file to be read.

    Returns:
        tuple: A tuple containing two elements:
            - data (numpy.ndarray): A NumPy array containing the data read from the CSV file.
            - df (pandas.DataFrame): A Pandas DataFrame created from the data with appropriate column names.
    """
    with open(filename,'rt') as raw_data:
        data=np.loadtxt(raw_data,delimiter=',')

    column_name=['Time','Event']
    #***
    for i in range(1046):
        column_name.append('Fea_'+str(i+1))

    #creating a data frame
    data = np.delete(data, 2, axis=1)
    df = pd.DataFrame(data = data, columns = column_name)

    return data,df



class CPU_Unpickler(pickle.Unpickler):
    """
    Custom unpickler to handle deserialization of objects serialized with PyTorch's torch.save() function
    using map_location='cpu'.

    This class extends pickle.Unpickler and overrides the find_class method to customize the loading
    behavior for a specific scenario.

    Usage:
        To use this custom unpickler, pass it to pickle.load() when loading pickled objects.

        Example:
        with open('serialized_data.pkl', 'rb') as file:
            custom_unpickler = CPU_Unpickler(file)
            deserialized_object = custom_unpickler.load()
    """

    def find_class(self, module: str, name: str):
        """
        Override the find_class method to customize the class loading behavior during unpickling.

        Args:
            module (str): The name of the module being imported during unpickling.
            name (str): The name of the class within the module being imported.

        Returns:
            type: The class object associated with the specified module and name.
        """
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def get_pkl_dic(folder: str) -> Dict[str, int]:
    """
    Retrieve information from pickled files in the specified folder and create a dictionary with unique keys.

    The function lists all files in the specified folder with the ".pkl" extension. It opens each pickled file,
    extracts various arguments from the content, and generates a unique key based on these arguments.
    The function then creates a dictionary with the unique keys and assigns a value of 0 to each key.

    Args:
        folder (str): The folder path containing pickled files with the ".pkl" extension.

    Returns:
        dict: A dictionary containing unique entries with keys generated from extracted arguments and values set to 0.
    """
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and ".pkl" in f]
    already = {}
    for file in onlyfiles:
        file_loc = join(folder,file)
        with open(file_loc,'rb') as f:
            res=CPU_Unpickler(f).load()

            args = res["args"]

            key = str(args.model) +"_"+ str(args.Test_domain_id) +"_"+str(args.target_domain_type) \
                +"_"+str(args.target_domain_weight)+"_"+str(args.percent_target) \
                +"_"+str(args.lr)+"_"+str(args.lr_alphas) \
                +"_"+str(args.norm_alphas)+ "_"+str(args.current_experiment)

            already[key] = 0
    return already

def split_source_target_survival(X_survival: List[np.ndarray], y_survival: List[np.ndarray], event_survival: List[np.ndarray],
                                 target: int, device: str = 'cuda:0', merge: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor,
                                                                                                 List[torch.Tensor], torch.Tensor,
                                                                                                 List[torch.Tensor], torch.Tensor]:
    """
    Split the survival data into source and target samples and convert them into PyTorch Tensors.

    Parameters:
        X_survival (List[np.ndarray]): List of numpy arrays containing the samples for each survival domain.
        y_survival (List[np.ndarray]): List of numpy arrays containing the survival times for each sample in each survival domain.
        event_survival (List[np.ndarray]): List of numpy arrays containing the event indicators for each sample in each sirvival domain.
        target (int): Index of the target domain to be separated from the source domains.
        device (str): Device to store the PyTorch Tensors, 'cuda:X' for GPU or 'cpu' for CPU (default is 'cuda:0').
        merge (bool): If True, merge all source domains into one, creating a single source survival domain (default is False).

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]: ***
        - X_s (List[torch.Tensor]): List of PyTorch Tensors containing the source survival domains.
        - X_t (torch.Tensor): PyTorch Tensor containing the samples of the target survival domain.
        - y_s (List[torch.Tensor]): List of PyTorch Tensors containing the survival times of the source survival domains.
        - y_t (torch.Tensor): PyTorch Tensor containing the survival times of the samples of the target survival domain.
        - event_s (List[torch.Tensor]): List of PyTorch Tensors containing the event indicators of the source survival domains.
        - event_t (torch.Tensor): PyTorch Tensor containing the event indicator of the target survival domain.
    """
    sources = np.where(np.arange(len(X_survival))!=target)[0]
    X_t = X_survival[target]
    y_t = y_survival[target] .squeeze()
    event_t = event_survival[target] .squeeze()

    if merge:
        X_s = [np.concatenate([X_survival[s] for s in sources])]
        y_s = [np.concatenate([y_survival[s] for s in sources])]
        event_s = [np.concatenate([event_survival[s] for s in sources])]
    else:
        X_s = [X_survival[s] for s in sources]
        y_s = [y_survival[s] for s in sources]
        event_s = [event_survival[s] for s in sources]

    X_s = [torch.Tensor(x).to(device) for x in X_s]
    X_t = torch.Tensor(X_t).to(device)
    y_s = [torch.Tensor(y).to(device).unsqueeze(1) for y in y_s]
    y_t = torch.Tensor(y_t).to(device).unsqueeze(1)

    event_s = [torch.Tensor(event).to(device).unsqueeze(1) for event in event_s]
    event_t = torch.Tensor(event_t).to(device).unsqueeze(1)


    return X_s, X_t, y_s, y_t, event_s, event_t

def batch_loader_survival_supervised_weighted(X_s: List[torch.Tensor], y_s: List[torch.Tensor], event_s: List[torch.Tensor],
                                              X_t: torch.Tensor, y_t: torch.Tensor, event_t: torch.Tensor,
                                              weight_supervised_t: float, device: str, batch_size: int = 64,
                                              shuffle: bool = True, random_state: int = 0) -> Generator[Tuple[List[torch.Tensor],
                                              List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
                                              List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]], None, None]:
    """
    Batch loader for supervised and unsupervised weighted survival data.
    The output contains two types of sets: xs1, ys1, eventsource1, weights1 and xs2, ys2, eventsource2, weights2. The first set includes
        batches only from the source domains and the second set includes batches from the source and target domains.

    Parameters:
        X_s (List[torch.Tensor]): List of PyTorch Tensors containing the samples of each survival source domain.
        y_s (List[torch.Tensor]): List of PyTorch Tensors containing the survival times of the samples of each survival source domain.
        event_s (List[torch.Tensor]): List of PyTorch Tensors containing the event indicators of the samples of each survival source domain.
        X_t (torch.Tensor): PyTorch Tensor containing the samples of the target survival domain.
        y_t (torch.Tensor): PyTorch Tensor containing the survival time of the samples of the target survival somain.
        event_t (torch.Tensor): PyTorch Tensor containing the event indicator of the samples of the target survival domain.
        weight_supervised_t (float): Weight for the supervised target samples. ***
        device (str): Device to store the PyTorch Tensors, 'cuda:X' for GPU or 'cpu' for CPU.
        batch_size (int): Batch size (default is 64).
        shuffle (bool): If True, shuffle the data for each batch (default is True).
        random_state (int): Random seed for reproducibility (default is 0).

    Yields:
        Generator[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
                    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]], None, None]:
        A generator that yields tuples of batch data, each containing the following tensors for each source.
        There are mainly two sets of outputs: xs1, ys1, eventsource1, weights1 and xs2, ys2, eventsource2, weights2. The first set includes
        batches only from the source domains and the second set includes batches from the source and target domains.
        - xs1 (List[torch.Tensor]): List of PyTorch Tensors containing the batches from the survival source domains.
        - ys1 (List[torch.Tensor]): List of PyTorch Tensors containing the survival times for the batches from the source survival domains.
        - eventsource1 (List[torch.Tensor]): List of PyTorch Tensors containing the event indicators for the batches from the source survival domains.
        - weights1 (List[torch.Tensor]): List of PyTorch Tensors containing weights for the batches from the source survival domains.
        - xs2 (List[torch.Tensor]): List of PyTorch Tensors containing the batches from the source and target survival domains combined.
        - ys2 (List[torch.Tensor]): List of PyTorch Tensors containing the survival times for the batches from the source and target survival domains combined.
        - eventsource2 (List[torch.Tensor]): List of PyTorch Tensors containing the event indicators for the batches from the source and target survival domains combined.
        - weights2 (List[torch.Tensor]): List of PyTorch Tensors containing weights for the batches from the source and target survival domains combined.
    """
    if random_state is not None:
        np.random.seed(random_state)
    inputs, targets, events = [x.clone() for x in X_s], [y.clone() for y in y_s], [events.clone() for events in event_s]

    input_sizes = [X_s[i].shape[0] for i in range(len(X_s))]
    max_input_size = max(input_sizes)
    n_sources = len(X_s)
    num_blocks = max(int(max_input_size / batch_size),1)

    #print(num_blocks)
    #pdb.set_trace()
    if shuffle:
        for i in range(n_sources):
            r_order = np.arange(input_sizes[i])
            np.random.shuffle(r_order)
            inputs[i], targets[i], events[i] = inputs[i][r_order, :], targets[i][r_order], events[i][r_order]
        t_orders = []

        for i in range(n_sources):
            order = np.random.choice(num_blocks,len(event_t))
            t_order = []
            for j in range(num_blocks):
                t_order += [np.where(order==j)[0]]
            t_orders += [t_order]

    for j in range(num_blocks):
        xs1, ys1, eventsource1, weights1, xs2, ys2, eventsource2, weights2 = [], [], [], [], [], [], [], []
        for i in range(n_sources):
            ridx = np.random.choice(input_sizes[i], batch_size)
            tindex = t_orders[i][j]

            #t_orders[i][j]
            xs1.append(inputs[i][ridx, :])
            ys1.append(targets[i][ridx])
            eventsource1.append(events[i][ridx])
            weights1.append(torch.ones(batch_size, device=device))

            xs2.append(torch.cat((inputs[i][ridx, :],X_t[tindex,:])))
            ys2.append(torch.cat((targets[i][ridx],y_t[tindex])))
            eventsource2.append(torch.cat((events[i][ridx],event_t[tindex])))
            weights2.append(torch.cat((torch.ones(batch_size, device=device),torch.tensor([weight_supervised_t], device=device).repeat(t_orders[i][j].shape[0]))))

        yield xs1, ys1, eventsource1, weights1, xs2, ys2, eventsource2, weights2

def c_index(risk_pred, y, e) -> float:
    """
    Calculate the concordance index (C-index) for a survival prediction model.

    The concordance index is a measure of how well the predicted risks (risk_pred) align with the actual
    observed outcomes (y) and event indicators (e).

    Args:
        risk_pred (array_like): Predicted risks for each sample.
        y (array_like or torch.Tensor): Observed survival times or event times (time-to-event outcomes).
        e (array_like or torch.Tensor): Event indicators, where 1 indicates an event occurred and 0 indicates censored data.

    Returns:
        float: The calculated concordance index (C-index).
    """
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)
