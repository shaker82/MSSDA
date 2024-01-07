"""
       <NAME OF THE PROGRAM THIS FILE BELONGS TO>

  File:     <SurviveSDI.py>
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
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from SurvAIve.MSSDA.utils.enums import Target_domain_type
import pdb

class SurviveSDI(nn.Module):
    """
    Multi-Source Survival Domain Adaptation method.
    See Shaker, Ammar, and Carolin Lawrence. "Multi-Source Survival Domain Adaptation."
    In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8, pp. 9752-9762. 2023.
    """
    def __init__(self, args):
        super().__init__()
        #super(MSDANetTRE, self).__init__()        
        self.input_dim = args["input_dim"]
        self.output_dim = args['output_dim']
        self.n_sources = args['n_sources']
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.feature_extractor = args['feature_extractor']
        # Parameter of the final regressor.
        print(args)
        self.predictor = args['prediction_head']
        self.discriminator = args['discriminator_head']
        self.min_y_value = args['min_y_value']
        self.max_y_value = args['max_y_value']
        self.norm_alphas = args['norm_alphas']
        self.logger = args['logger']

        #Parameter
        #self.register_parameter(name='alpha', param=torch.nn.Parameter(torch.Tensor(np.ones(self.n_sources)/self.n_sources)) )
        self.source_weights = torch.nn.Parameter(torch.Tensor(np.ones(self.n_sources) / self.n_sources))
        self.consider_hidden_feat = [i for i in range(len(self.feature_extractor)) if  "ELU" in str(self.feature_extractor[i])]

    def set_optimizers(self,
                       feature_optimizer: optim.Adam, predictor_optimizer: optim.Adam,
                       discriminator_optimizer: optim.Adam, alpha_optimizer: optim.Adam
                       ) -> None:
        """
        Configures optimizers for each class of parameters.

        Args:
            feature_optimizer (torch.optim.Adam): The optimizer for feature extrator's parameters.
            predictor_optimizer (torch.optim.Adam): The optimizer for the predictor's parameters.
            discriminator_optimizer (torch.optim.Adam): The optimizer for discriminator's parameters.
            alpha_optimizer (torch.optim.Adam): The optimizer for the alphas, the weights for the source domains.

        Returns:
            None
        """
        self.feature_optimizer = feature_optimizer
        self.predictor_optimizer = predictor_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.alpha_optimizer = alpha_optimizer

        
    def zero_grads(self) -> None:
        """
        Set all gradients to zero.

        Args:
            None

        Returns:
            None
        """
        self.feature_optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

    def forward(self,
        X_s: List[torch.Tensor], X_t: torch.Tensor
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            X_s (List[torch.Tensor]): list of torch.Tensor (m_s, d), source domains.
            X_t (torch.Tensor): torch.Tensor (n, d), target data.

        Returns:
            y_source_prediction (List[torch.Tensor]): list of torch.Tensor (m_s), source domains' predictions.
            y_source_discrepancy (List[torch.Tensor]): list of torch.Tensor (m_s), source domain's predictions.
            y_target_prediction (torch.Tensor): torch.Tensor, target domain prediction.
            y_target_discrepancy (torch.Tensor): torch.Tensor, target domain prediction.
        """
        # Feature extractor
        sx, tx = X_s.copy(), X_t.clone()
        for i in range(self.n_sources):
            for hidden in self.feature_extractor:
                sx[i] = hidden(sx[i])
        for hidden in self.feature_extractor:
            tx = hidden(tx)
            
        # Predictor head
        y_source_prediction = []
        for i in range(self.n_sources):
            y_sx = sx[i].clone()
            for hidden in self.predictor:
                y_sx = hidden(y_sx)
            y_source_prediction.append(self.clamp(y_sx))
    
        y_tx = tx.clone()
        for hidden in self.predictor:
            y_tx = hidden(y_tx)
        y_target_prediction = self.clamp(y_tx)
            
        # Discriminator head
        y_source_discrepancy = []
        for i in range(self.n_sources):
            y_tmp = sx[i].clone()
            for hidden in self.discriminator:
                y_tmp = hidden(y_tmp)
            y_source_discrepancy.append(self.clamp(y_tmp))
        y_tmp = tx.clone()
        for hidden in self.discriminator:
            y_tmp = hidden(y_tmp)
        y_target_discrepancy = self.clamp(y_tmp)
        return y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy

    def forward_full(self,
        X_s: List[torch.Tensor],
        X_t: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[List[torch.Tensor]], List[torch.Tensor]]:
        """
        Full forward pass with full output including the outputs of the hidden layers.

        Args:
            X_s (List[torch.Tensor]): list of torch.Tensor (m_s, d), source domains.
            X_t (torch.Tensor): torch.Tensor (n, d), target data.

        Returns:
            y_source_prediction (List[torch.Tensor]): list of torch.Tensor (m_s), source domains' predictions.
            y_source_discrepancy (List[torch.Tensor]): list of torch.Tensor (m_s), source domain's predictions.
            y_target_prediction (torch.Tensor): torch.Tensor, target domain prediction.
            y_target_discrepancy (torch.Tensor): torch.Tensor, target domain prediction.
            All_sx (List[List[torch.Tensor]]): list of lists of torch.Tensor, for each domain and each hidden layer.
            All_tx (List[torch.Tensor]): list of torch.Tensor, for each hidden layer for the target domain.
        """
        # Feature extractor        
        sx, tx = X_s.copy(), X_t.clone()
        All_sx = [[None for i in range(self.n_sources)] for hidden in self.consider_hidden_feat]
        All_tx = [None for hidden in self.consider_hidden_feat]        
        for i in range(self.n_sources):
            temp = 0
            for j,hidden in enumerate(self.feature_extractor):
                sx[i] = hidden(sx[i])
                if j in self.consider_hidden_feat:
                    All_sx[temp][i] = sx[i]
                    temp +=1
        temp = 0
        for j,hidden in enumerate(self.feature_extractor):
            tx = hidden(tx)            
            if j in self.consider_hidden_feat:
                All_tx[temp] = tx
                temp +=1
            
        # Predictor head
        y_source_prediction = []
        for i in range(self.n_sources):
            y_sx = sx[i].clone()
            for hidden in self.predictor:
                y_sx = hidden(y_sx)
            y_source_prediction.append(self.clamp(y_sx))
    
        y_tx = tx.clone()
        for hidden in self.predictor:
            y_tx = hidden(y_tx)
        y_target_prediction = self.clamp(y_tx)
            
        # Discriminator head
        y_source_discrepancy = []
        for i in range(self.n_sources):
            y_tmp = sx[i].clone()
            for hidden in self.discriminator:
                y_tmp = hidden(y_tmp)
            y_source_discrepancy.append(self.clamp(y_tmp))
        y_tmp = tx.clone()
        for hidden in self.discriminator:
            y_tmp = hidden(y_tmp)
        y_target_discrepancy = self.clamp(y_tmp)
        return y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, All_sx, All_tx

    def train_predictor(self,
                        x_bs1: List[torch.Tensor], x_bs2: List[torch.Tensor], x_bt: torch.Tensor,
                        y_bs1: List[torch.Tensor], y_bs2: List[torch.Tensor],
                        event_bs1: List[torch.Tensor], event_bs2: List[torch.Tensor], event_bt: torch.Tensor,
                        weight_bs1: List[torch.Tensor], weight_bs2: List[torch.Tensor],
                        largeVal: float, clip: float=1, pred_only: bool=False) -> None:
        """
        Train feature extractor using the weighted negative loglikelihood loss. Depending on the last argument,
        the predictor head could also be trained.
        The training uses the first set of samples, x_bs1, y_bs1, and event_bs1; these are used to train only
        the weighted negative loglikelihood loss. This data could be only from the source domains,
        or from the source domains and a portion of labeled target domain.
        The second set of samples (not used), x_bs1, y_bs1, and event_bs1; these are used to train only
        divergence loss. This data could be only from the source domains, or from the source domains
        and a portion of labeled target domain.
        Args:
            x_bs1 (List[torch.Tensor]): Input data used to train only the weighted negative loglikelihood loss.
            x_bs2 (List[torch.Tensor]): Input data used to train only divergence loss.
            x_bt (torch.Tensor): Unlabeled data from the target domain.
            y_bs1 (List[torch.Tensor]): Labels used to train only the weighted negative loglikelihood loss.
            y_bs2 (List[torch.Tensor]): Labels used to train only divergence loss.
            event_bs1 (List[torch.Tensor]): Event data used to train only the weighted negative loglikelihood loss.
            event_bs2 (List[torch.Tensor]): Event data used to train only divergence loss.
            event_bt (torch.Tensor): Events from the target domain.
            weight_bs1 (List[torch.Tensor]): Weights for samples used to train only the weighted negative loglikelihood loss.
            weight_bs2 (List[torch.Tensor]): Weights for samples used to train only divergence loss.
            largeVal (float): A large value used in the calculations.
            clip (float): Clipping parameter.
            pred_only (bool): a weight parameter for the weighted loss on the source domains.

        Returns:
            None.
        """
        #Training
        self.train()
        #Training on the sources
        y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy = self.forward(x_bs1, x_bt)
        y_s = [y_bs1[i].view(-1) for i in range(len(y_bs1))]
        y_source_prediction = [y_source_prediction[i].view(-1) for i in range(len(y_source_prediction))]
        event_bs1 = [event_bs1[i].view(-1) for i in range(len(event_bs1))]
        loss_pred = self.weighted_NegativeLogLikelihood(y_s, y_source_prediction, event_bs1, weight_bs1, self.source_weights)

        self.zero_grads()
        loss_pred.backward(retain_graph=False)
        #Gradients clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        #Optimization
        self.predictor_optimizer.step()
        if not pred_only:
            self.feature_optimizer.step()
        self.zero_grads()

    def train_discriminator(
        self, x_bs1: List[torch.Tensor], x_bs2: List[torch.Tensor], x_bt: torch.Tensor,
        y_bs1: List[torch.Tensor], y_bs2: List[torch.Tensor],
        event_bs1: List[torch.Tensor], event_bs2: List[torch.Tensor], event_bt: torch.Tensor,
        weight_bs1: List[torch.Tensor], weight_bs2: List[torch.Tensor],
        largeVal: float, clip: float=1.0)  -> None:
        """
        Train the discriminator to maximize the discrepancy.
        The first set of samples, x_bs1, y_bs1, and event_bs1; these are used to train only the weighted negative loglikelihood loss. This data could be only 
                from the source domains, or from the source domains and a portion of labeled target domain.
        The second set of samples, x_bs1, y_bs1, and event_bs1; these are used to train only divergence loss. This data could be only 
                from the source domains, or from the source domains and a portion of labeled target domain.
        Args:
            x_bs1 (list[torch.Tensor]): Input data used to train only the weighted negative loglikelihood loss.
            x_bs2 (list[torch.Tensor]): Input data used to train only divergence loss.
            x_bt (torch.Tensor): Unlabeled data from the target domain.
            y_bs1 (list[torch.Tensor]): Labels used to train only the weighted negative loglikelihood loss.
            y_bs2 (list[torch.Tensor]): Labels used to train only divergence loss.
            event_bs1 (list[torch.Tensor]): Event data used to train only the weighted negative loglikelihood loss.
            event_bs2 (list[torch.Tensor]): Event data used to train only divergence loss.
            event_bt (torch.Tensor): Event data for the target domain.
            weight_bs1 (list[torch.Tensor]): Weights for samples used to train only the weighted negative loglikelihood loss.
            weight_bs2 (list[torch.Tensor]): Weights for samples used to train only divergence loss.
            largeVal (float): A large value used in the calculations.
            clip (float): Clipping parameter.
        
        Returns:
            None.
        """
        self.train()
        device = x_bs2[0].device

        #Training the discriminator
        y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, All_sx, All_tx = self.forward_full(x_bs2, x_bt)

        y_bs2 = [y_bs2[i]/max(y_bs2[i]) for i in range(len(y_bs2))]
        #SDI
        res,sdi_sources,cond_div_t = self.compute_predictor_discrepancy_SDI(x_bs2, x_bt, y_bs2, y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, event_bs2, event_bt, weight_bs2, self.source_weights, largeVal)
        
        loss_disc = -res
        self.zero_grads()
        loss_disc.backward(retain_graph=False)
        #Gradients clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)

        ##Correction
        for name, param in self.named_parameters():
            if param.requires_grad:
                if torch.any(torch.isnan(param.grad)):
                    #pdb.set_trace()                    
                    self.logger.info("---------------------")
                    self.logger.info("train_discriminator:")
                    self.logger.info(name+" "+str(param)+" "+str(param.grad))
                    self.zero_grads()
                    return
        ###Correction
        
        #Optimization step
        self.discriminator_optimizer.step()
        self.zero_grads()

    def train_feat_extractor(
        self, x_bs1: List[torch.Tensor], x_bs2: List[torch.Tensor], x_bt: torch.Tensor,
        y_bs1: List[torch.Tensor], y_bs2: List[torch.Tensor],
        event_bs1: List[torch.Tensor], event_bs2: List[torch.Tensor], event_bt: torch.Tensor,
        weight_bs1: List[torch.Tensor], weight_bs2: List[torch.Tensor],
        largeVal: float, clip: float=1.0, mu: float =1)  -> None:

        """
        Train feature extractor using two different losses and two different source sets x_bs1 and x_bs2.
        The first set of samples, x_bs1, y_bs1, and event_bs1, are used to train only the weighted negative
        loglikelihood loss. This data could be only from the source domains, or from the source domains
        and a portion of labeled target domain.
        The second set of samples, x_bs1, y_bs1, and event_bs1, are used to train only divergence loss.
        This data could be only from the source domains, or from the source domains and a portion of
        labeled target domain.
        Args:
            x_bs1 (List[torch.Tensor]): Input data used to train only the weighted negative loglikelihood loss.
            x_bs2 (List[torch.Tensor]): Input data used to train only divergence loss.
            x_bt (torch.Tensor): Unlabeled data from the target domain.            
            y_bs1 (List[torch.Tensor]): Labels used to train only the weighted negative loglikelihood loss.
            y_bs2 (List[torch.Tensor]): Labels used to train only divergence loss.
            event_bs1 (List[torch.Tensor]): Event data used to train only the weighted negative loglikelihood loss.
            event_bs2 (List[torch.Tensor]): Event data used to train only divergence loss.
            event_bt (torch.Tensor): Events for the target domain.
            weight_bs1 (List[torch.Tensor]): Weights for samples used to train only the weighted negative
            weight_bs2 (List[torch.Tensor]): Weights for samples used to train only divergence loss.
            largeVal (float): A large value used in the calculations.
            clip (float): Clipping parameter.
            mu (float): a weight parameter for the weighted loss on the source domains.

        Returns:
            None.
        """
        #Training
        self.train()
        device = x_bs1[0].device

        #Feature training
        y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, All_sx, All_tx = self.forward_full(x_bs1, x_bt)

        y_s_1 = [y_bs1[i].view(-1) for i in range(len(y_bs1))]
        y_spred_1 = [y_source_prediction[i].view(-1) for i in range(len(y_source_prediction))]
        events_s_1 = [event_bs1[i].view(-1) for i in range(len(event_bs1))]
        source_loss = self.weighted_NegativeLogLikelihood(y_s_1, y_spred_1, events_s_1, weight_bs1, self.source_weights)

        #Calling again the forward of the data provided for the divergence
        y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, All_sx, All_tx = self.forward_full(x_bs2, x_bt)
        y_bs2 = [y_bs2[i]/max(y_bs2[i]) for i in range(len(y_bs2))]
        #SDI            
        res,sdi_sources,cond_div_t = self.compute_feat_discrepancy_SDI(x_bs2, x_bt, y_bs2, y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, event_bs2, event_bt, weight_bs2, self.source_weights, largeVal)
        loss = res + source_loss*mu

        self.zero_grads()
        loss.backward(retain_graph=False)
        #Gradients clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        
        ###Correction
        for name, param in self.named_parameters():
            if param.requires_grad:
                if torch.any(torch.isnan(param.grad)):                    
                    self.logger.info("---------------------")
                    self.logger.info("train_feat_extractor:")
                    self.logger.info(name+" "+str(param)+" "+str(param.grad))
                    self.zero_grads()
                    return
        ###Correction
        
        #Optimization step
        self.feature_optimizer.step()
        self.zero_grads()
        
    def train_source_weights(self,
                             x_bs1: List[torch.Tensor], x_bs2: List[torch.Tensor], x_bt: torch.Tensor,
                             y_bs1: List[torch.Tensor], y_bs2: List[torch.Tensor],
                             event_bs1: List[torch.Tensor], event_bs2: List[torch.Tensor], event_bt: torch.Tensor,
                             weight_bs1: List[torch.Tensor], weight_bs2: List[torch.Tensor],
                             largeVal: float, clip: float=1, lam_alpha: float=0.01) -> None:

        """
        Train alpha vector that weights the source domains.
        The training uses the first set of samples, x_bs1, y_bs1, and event_bs1; these are used to train only
        the weighted negative loglikelihood loss. This data could be only from the source domains,
        or from the source domains and a portion of labeled target domain.
        The second set of samples (not used), x_bs1, y_bs1, and event_bs1; these are used to train only
        divergence loss. This data could be only from the source domains, or from the source domains
        and a portion of labeled target domain.
        Args:
            x_bs1 (List[torch.Tensor]): Input data used to train only the weighted negative loglikelihood loss.
            x_bs2 (List[torch.Tensor]): Input data used to train only divergence loss.
            x_bt (torch.Tensor): Unlabeled data from the target domain.
            y_bs1 (List[torch.Tensor]): Labels used to train only the weighted negative loglikelihood loss.
            y_bs2 (List[torch.Tensor]): Labels used to train only divergence loss.
            event_bs1 (List[torch.Tensor]): Event data used to train only the weighted negative loglikelihood loss.
            event_bs2 (List[torch.Tensor]): Event data used to train only divergence loss.
            event_bt (torch.Tensor): Event data for the target domain.
            weight_bs1 (List[torch.Tensor]): Weights for samples used to train only the weighted negative
            weight_bs2 (List[torch.Tensor]): Weights for samples used to train only divergence loss.
            largeVal (float): A large value used in the calculations.
            clip (float): Clipping parameter.
            lam_alpha (float): a weight parameter for norm of the alphas vector.

        Returns:
            None.
        """
        #Training
        self.train()     
        device = x_bs2[0].device
        #Feature training
        y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, All_sx, All_tx = self.forward_full(x_bs2, x_bt)
        y_bs2 = [y_bs2[i]/max(y_bs2[i]) for i in range(len(y_bs2))]

        #SDI            
        res,sdi_sources,cond_div_t = self.compute_alpha_discrepancy_SDI(x_bs2, x_bt, y_bs2, y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, event_bs2, event_bt, weight_bs2, self.source_weights, largeVal)
        loss_disc = res + lam_alpha*torch.norm(self.source_weights, p=self.norm_alphas)
        self.zero_grads()
        loss_disc.backward(retain_graph=False)
        #Gradients clipping
        torch.nn.utils.clip_grad_norm_(self.source_weights, clip)
        
        ###Correction
        for name, param in self.named_parameters():
            if param.requires_grad:
                if torch.any(torch.isnan(param.grad)):                 
                    self.logger.info("---------------------")
                    self.logger.info("train_source_weights:")
                    self.logger.info(name+" "+str(param)+" "+str(param.grad))
                    self.zero_grads()
                    return
        ###Correction
        #Optimization step
        self.alpha_optimizer.step()
        self.zero_grads()
        #Normalization (||alpha||_1=1)
        with torch.no_grad():
            self.source_weights.clamp_(1 / (self.n_sources * 10), 1 - 1 / (self.n_sources * 10))
            self.source_weights.div_(torch.norm(F.relu(self.source_weights), p=1))
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output for the given input tensor.

        Args:
            X (torch.Tensor): The input tensor for prediction.

        Returns:
            torch.Tensor: The predicted output tensor.
        """        
        z = X.clone()
        for hidden in self.feature_extractor:
            z = hidden(z)
        for hidden in self.predictor:
            z = hidden(z)
        return self.clamp(z)
    
    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamps the input tensor element-wise between `min_pred` and `max_pred` values.

        Args:
            x (torch.Tensor): The input tensor to be clamped.

        Returns:
            torch.Tensor: The clamped tensor.
        """
        return torch.clamp_(x, self.min_y_value, self.max_y_value)

    def compute_all_losses(self,
                           X_s: List[torch.Tensor], X_t: torch.Tensor, y_s: List[torch.Tensor],
                           events_s: List[torch.Tensor], events_t: torch.Tensor,
                           largeVal: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Compute the losses
        Args:
            X_s (List[torch.Tensor]): Input data for the source domains.
            X_t (torch.Tensor): Input data for the target domain.
            y_s (List[torch.Tensor]): Labels for the source domains.
            events_s (List[torch.Tensor]): Event data for the source domain.
            events_t (torch.Tensor): Event data for the target domain.
            largeVal (int): A large value used in the calculations.
            device (string): the device name.
       
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The nagative loglikelihood los on the source domains, and the discrepancy between source and target domains based on the SDI.

        """
        y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy = self.forward(X_s, X_t)

        y_s_loss = [y_s[i].view(-1) for i in range(len(y_s))]
        y_spred_loss = [y_source_prediction[i].view(-1) for i in range(len(y_source_prediction))]
        events_s_loss = [events_s[i].view(-1) for i in range(len(events_s))]        

        Weights = [torch.ones(events_s[i].shape[0], device=device) for i in range(len(events_s))]
        source_loss = self.weighted_NegativeLogLikelihood(y_s_loss, y_spred_loss, events_s_loss, Weights, self.source_weights)

        #SDI
        disc2_SDI,weightd_cond_div_s_SDI,cond_div_t_SDI = self.compute_predictor_discrepancy_SDI(X_s, X_t, y_s, y_source_prediction, y_source_discrepancy, y_target_prediction, y_target_discrepancy, events_s, events_t, Weights, self.source_weights, largeVal)

        return source_loss, disc2_SDI

    def compute_predictor_discrepancy_SDI(
        self,
        X_s: List[torch.Tensor], X_t: torch.Tensor,
        y_s: List[torch.Tensor],
        ranking_source_prediction: List[torch.Tensor], ranking_source_discrepancy: List[torch.Tensor],
        ranking_target_prediction: torch.Tensor, ranking_target_discrepancy: torch.Tensor,
        events_s: List[torch.Tensor], events_t: torch.Tensor,
        weights_s: List[torch.Tensor], alpha: torch.nn.parameter.Parameter, largeVal: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Computes the discrepancy between a source domain (X_s, y_s) and a target domain (X_t) based on the difference on the 
        SDI (symmetric discordance index). The SDI is computed between the predictor, and the adversarial predictor.
        This function is used to train the adversarial predictor.
        
        Args:
            X_s (List[torch.Tensor]): Input data for the source domain.
            X_t (torch.Tensor): Input data for the target domain.
            y_s (List[torch.Tensor]): Labels for the source domain.
            ranking_source_prediction (List[torch.Tensor]): Rankings related to the source domains' predictions using predictor.
            ranking_source_discrepancy (List[torch.Tensor]): Rankings related to the source domains' predictions using adversarial predictor.
            ranking_target_prediction (torch.Tensor): Rankings related to the target domain predictions using predictor.
            ranking_target_discrepancy (torch.Tensor): Rankings related to the target domain predictions using adversarial predictor.
            events_s (List[torch.Tensor]): Event data for the source domains.
            events_t (torch.Tensor): Event data for the target domain.
            weights_s (List[torch.Tensor]): Weights for the source domains samples (sample-wise)
            alpha (torch.nn.parameter.Parameter): Weights for the source domains (domain-wise).
            largeVal (int): A large value used in the calculations.
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: The discrepancy between source and target domains, sdi_sources, and sdi_target.
        """
        sdi_sources_weighted, sdi_sources = self.compute_wighted_source_SDI(
            X_s, ranking_source_prediction, ranking_source_discrepancy, events_s, weights_s, alpha, largeVal
        )
        sdi_target = self.compute_target_SDI(
            X_t, ranking_target_prediction, ranking_target_discrepancy, events_t, largeVal
        )
        return torch.abs(sdi_sources_weighted - sdi_target), sdi_sources, sdi_target

    def compute_feat_discrepancy_SDI(
        self,
        X_s: List[torch.Tensor], X_t: torch.Tensor, y_s: List[torch.Tensor],
        ranking_source_prediction: List[torch.Tensor], ranking_source_discrepancy: List[torch.Tensor],
        ranking_target_prediction: torch.Tensor, ranking_target_discrepancy: torch.Tensor,
        events_s: List[torch.Tensor], events_t: torch.Tensor,
        weights_s: List[torch.Tensor], alpha: torch.Tensor, largeVal: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Computes the discrepancy between a source domain (X_s, y_s) and a target domain (X_t) based on the difference on the 
        SDI (symmetric discordance index). The SDI is computed between the predictor, and the adversarial predictor.
        This function is used to train the feature extractor.
        Args:
            X_s (List[torch.Tensor]): Input data for the source domain.
            X_t (torch.Tensor): Input data for the target domain.
            y_s (List[torch.Tensor]): Labels for the source domain.
            ranking_source_prediction (List[torch.Tensor]): Rankings related to the source domains' predictions using predictor.
            ranking_source_discrepancy (List[torch.Tensor]): Rankings related to the source domains' predictions using adversarial predictor.
            ranking_target_prediction (torch.Tensor): Rankings related to the target domain predictions using predictor.
            ranking_target_discrepancy (torch.Tensor): Rankings related to the target domain predictions using adversarial predictor.
            events_s (List[torch.Tensor]): Event data for the source domain.
            events_t (torch.Tensor): Event data for the target domain.
            weights_s (List[torch.Tensor]): Weights for the source domains samples (sample-wise).
            alpha (torch.Tensor): Weights for the source domains (domain-wise).
            largeVal (int): A large value used in the calculations.
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: The discrepancy between source and target domains, sdi_sources, and sdi_target.
        """
        sdi_sources_weighted,sdi_sources = self.compute_wighted_source_SDI(X_s, ranking_source_prediction, ranking_source_discrepancy, events_s, weights_s, alpha, largeVal)
        sdi_target = self.compute_target_SDI(X_t, ranking_target_prediction, ranking_target_discrepancy, events_t, largeVal)
        return torch.abs(sdi_sources_weighted -  sdi_target),sdi_sources,sdi_target

    def compute_alpha_discrepancy_SDI(
        self,
        X_s: List[torch.Tensor], X_t: torch.Tensor, y_s: List[torch.Tensor],
        ranking_source_prediction: List[torch.Tensor], ranking_source_discrepancy: List[torch.Tensor],
        ranking_target_prediction: torch.Tensor, ranking_target_discrepancy: torch.Tensor,
        events_s: List[torch.Tensor], events_t: torch.Tensor,
        weights_s: List[torch.Tensor], alpha: torch.Tensor, largeVal: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Computes the discrepancy between a source domain (X_s, y_s) and a target domain (X_t) based on the difference on the 
        SDI (symmetric discordance index). The SDI is computed between the predictor, and the adversarial predictor.
        This function is used to train the weights of the sources.
        
        Args:
            X_s (List[torch.Tensor]): Input data for the source domain.
            X_t (torch.Tensor): Input data for the target domain.
            y_s (List[torch.Tensor]): Labels for the source domain.
            ranking_source_prediction (List[torch.Tensor]): Rankings related to the source domains' predictions using predictor.
            ranking_source_discrepancy (List[torch.Tensor]): Rankings related to the source domains' predictions using adversarial predictor.
            ranking_target_prediction (torch.Tensor): Rankings related to the target domain predictions using predictor.
            ranking_target_discrepancy (torch.Tensor): Rankings related to the target domain predictions using adversarial predictor.
            events_s (List[torch.Tensor]): Event data for the source domain.
            events_t (torch.Tensor): Event data for the target domain.
            weights_s (List[torch.Tensor]): Weights for the source domain.
            largeVal (int): A large value used in the calculations.
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: The discrepancy between source and target domains, sdi_sources, and sdi_target.
        """
        sdi_sources_weighted,sdi_sources = self.compute_wighted_source_SDI(X_s, ranking_source_prediction, ranking_source_discrepancy, events_s, weights_s, alpha, largeVal)
        sdi_target = self.compute_target_SDI(X_t, ranking_target_prediction, ranking_target_discrepancy, events_t, largeVal)
        return torch.abs(sdi_target - sdi_sources_weighted),sdi_sources,sdi_target

    def compute_target_SDI(
        self,
        X_t: torch.Tensor,
        ranking1: torch.Tensor, ranking2: torch.Tensor,
        events_t: torch.Tensor, largeVal: int
    ) -> torch.Tensor:
        """
        Computes SDI (symmetric discordance index) on the target domain between rankings computed by the predictor and the adversarial predictor.
        
        Args:
            X_t (torch.Tensor): Input data for the target domain.
            ranking1 (torch.Tensor): Ranking 1 on the target domain computed by predictor h.
            ranking2 (torch.Tensor): Ranking 2 on the target domain computed by the adversarial predictor.
            events_t (torch.Tensor): Event data for the target domain.
            largeVal (int): A large value used in the calculations.
        
        Returns:
            torch.Tensor: The SDI on the target domain.
        """
        weights_t = torch.ones(events_t.shape[0], device=X_t.device)
        sdi_target, part1, part2, alpha1, alpha2 = self.SDI(ranking1, ranking2, events_t, weights_t, largeVal)
        return sdi_target

    def compute_wighted_source_SDI(
        self,
        X_s: torch.Tensor,
        ranking1: torch.Tensor, ranking2: torch.Tensor,
        events_s: torch.Tensor, weights_s: torch.Tensor,
        alpha: torch.nn.parameter.Parameter,
        largeVal: int

    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes the weighted SDI (symmetric discordance index) on the source domains between rankings computed by the predictor and the adversarial predictor.
        
        Args:
            X_s (List[torch.Tensor]): Input data for the source domain.
            ranking1 (List[torch.Tensor]): Ranking 1 on the source domains computed by predictor h.
            ranking2 (List[torch.Tensor]): Ranking 2 on the source domains computed by the adversarial predictor.
            events_s (List[torch.Tensor]): Event data for the source domains.
            weights_s (List[torch.Tensor]): Weights for the source domains samples (sample-wise).
            alpha (torch.nn.parameter.Parameter): Weights for the source domains (domain-wise).
            largeVal (int): A large value used in the calculations.
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The weighted SDI one the source domains, and the list of individual SDI values.
        """
        sdi_sources = []
        sdi_sources_weighted = 0

        for i in range(self.n_sources):
            value, part1, part2, alpha1, alpha2 = self.SDI(ranking1[i], ranking2[i], events_s[i], weights_s[i], largeVal)
            sdi_sources_weighted += alpha[i] * value
            sdi_sources.append(value)
        
        return sdi_sources_weighted, sdi_sources

    def SDI(self, 
        ranking1: torch.Tensor, ranking2: torch.Tensor, 
        events: torch.Tensor, weights: torch.Tensor, largeVal: float
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the SDI (symmetric discordance index) metric between rankings.

        Args:
            ranking1 (torch.Tensor): The first ranking tensor.
            ranking2 (torch.Tensor): The second ranking tensor.
            events (torch.Tensor): The events tensor.
            weights (torch.Tensor): The weights tensor.
            largeVal (float): A large value used in the calculations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                the computed SDI, part1, part2, alpha1, alpha2 

        """
        risk_pred1 = ranking1.view(-1)
        risk_pred2 = ranking2.view(-1)
        events = events.view(-1)

        device = ranking1.device
        
        # The mask of dead samples
        died_mask  = events.view(-1).bool()

        # Predicted risks and weight of dead samples
        risk_pred1_event = risk_pred1[died_mask]
        risk_pred2_event = risk_pred2[died_mask]
        weight_event = weights[died_mask]

        # Predicted risks and weight of censored samples
        risk_pred1_censored = risk_pred1[~died_mask]
        risk_pred2_censored = risk_pred2[~died_mask]
        weight_censored = weights[~died_mask]

        # Initializing loss components
        margin_loss = torch.nn.MarginRankingLoss(1,reduction='none')
        part1 = torch.tensor(0.0, device=device)
        Z = torch.tensor(0.0, device=device)
        alpha1 = torch.tensor(0.0, device=device)
        alpha2 = torch.tensor(0.0, device=device)
        
        # Computing the loss component for each dead sample
        for i in range(risk_pred1_event.shape[0]-1):            
            v = largeVal*(risk_pred1_event[i]-risk_pred1_event[i+1:])*(risk_pred2_event[i]-risk_pred2_event[i+1:])
            res = margin_loss(v.exp(),torch.tensor([0.0],device = device),torch.tensor([1.0],device = device)) * weight_event[i+1:]            
            part1 +=res.sum()
            Z += weight_event[i+1:].sum()*weight_event[i]
            alpha1 += weight_event[i+1:].sum()*weight_event[i]
        part1 =part1/Z if Z>0 else torch.tensor(0.0, device=device)
        #print(Z,part1)
        
        # Computing the second loss component for each censored sample
        # Based on the Jaccard distance which is a measure.
        #          (|A\B|+|B\A|) 
        # J(A,B) = ------------- 
        #            (|A U B|)   
        
        part2 = torch.tensor(0.0, device=device)
        res_all = torch.tensor(0.0, device=device)
        for i in range(len(risk_pred1_censored)):
            #|A\B|+|B\A|
            v = largeVal*(risk_pred1_censored[i]-risk_pred1_event)*(risk_pred2_censored[i]-risk_pred2_event)
            res = margin_loss(v.exp(),torch.tensor([0.0],device = device),torch.tensor([1.0],device = device)) * weight_event * weight_censored[i]
            res = res.sum()
            #|A|
            v1 = largeVal*(risk_pred1_censored[i]-risk_pred1_event)
            res1 = margin_loss(v1.exp(),torch.tensor([0.0],device = device),torch.tensor([1.0],device = device)) * weight_event * weight_censored[i]
            res1 = res1.sum()
            #|B|
            v2 = largeVal*(risk_pred2_censored[i]-risk_pred2_event)
            res2 = margin_loss(v2.exp(),torch.tensor([0.0],device = device),torch.tensor([1.0],device = device)) * weight_event * weight_censored[i]
            res2 = res2.sum()
            #|A U B| = (|A|+|B| + |A\B|+|B\A|)/2
            res_all = (res1 + res2 + res)/2
            res_final = res/res_all
            #print(res_all)
            alpha2 += (weight_event * weight_censored[i]).sum()
            if res_all>0:
                part2 +=res_final
        alpha2 /=2        
        part2 = part2 /len(risk_pred1_censored) if res_all>0 else 0
        
        # Applying the weighted avarage on the two components
        result =  (alpha1*part1+alpha2*part2)/(alpha1+alpha2)
        result = result.float()
        return result, part1, part2, alpha1, alpha2

    def NegativeLogLikelihood(self, event_time: torch.Tensor, risk_pred: torch.Tensor, events: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative log-likelihood loss for survival analysis.

        Args:
            event_time (torch.Tensor): Tensor containing the event times.
            risk_pred (torch.Tensor): Tensor containing the predicted risks.
            events (torch.Tensor): Tensor indicating whether an event occurred (1) or not (0).
            weights (torch.Tensor): Tensor indicating the weights of the samples.

        Returns:
            torch.Tensor: Negative log-likelihood loss.

        Raises:
            ValueError: If the input tensors have inconsistent shapes.

        """
        if event_time.shape != risk_pred.shape or event_time.shape != events.shape:
            raise ValueError("Input tensors must have the same shape.")

        # Sort the tensors based on event_time in descending order
        sorted_indices = event_time.sort(descending=True)[1]
        events = events[sorted_indices]
        risk_pred = risk_pred[sorted_indices]

        events = events.float()

        # Compute uncensored likelihood
        uncensored_likelihood = weights*risk_pred - (risk_pred.exp()*weights).cumsum(0).log()

        # Compute censored likelihood
        censored_likelihood = uncensored_likelihood * events

        # Compute the number of observed events
        num_observed_events = events.sum()

        # Compute negative log-likelihood
        neg_likelihood = -censored_likelihood.sum() / num_observed_events

        return neg_likelihood

    def init(self, args, device):
        self.largeVal = args.largeVal
        self.target_domain_type = Target_domain_type(args.target_domain_type)
        self.batch_size = args.batch_size
        self.device = device

        self.lr, self.lr_alphas, self.batch_size, self.bottleneck_size = args.lr, args.lr_alphas, args.batch_size, args.bottleneck_size
        opt_feat = torch.optim.Adam([{'params': self.feature_extractor.parameters()}],lr=self.lr)
        opt_pred = torch.optim.Adam([{'params': self.predictor.parameters()}], lr=self.lr)
        opt_disc = torch.optim.Adam([{'params': self.discriminator.parameters()}], lr=self.lr)
        opt_alpha = torch.optim.Adam([{'params': self.source_weights}], lr=self.lr_alphas)
        self.set_optimizers(opt_feat, opt_pred, opt_disc, opt_alpha)

    def run_epoch(self, loader, X_t, event_t):
        # xs1, ys1, eventsource1, and weights1 include only data of the source domains.
        # xs2, ys2, eventsource2, and weights2 include the source domains data and a percentage from the target domain.
        for xs1, ys1, eventsource1, weights1, xs2, ys2, eventsource2, weights2 in loader:

            if self.target_domain_type==Target_domain_type.no_target_learning:
                x_bs1, y_bs1, event_bs1, weight_bs1 = xs1, ys1, eventsource1, weights1
                x_bs2, y_bs2, event_bs2, weight_bs2 = xs1, ys1, eventsource1, weights1
            elif self.target_domain_type==Target_domain_type.target_only_in_loss:
                x_bs1, y_bs1, event_bs1, weight_bs1 = xs2, ys2, eventsource2, weights2
                x_bs2, y_bs2, event_bs2, weight_bs2 = xs1, ys1, eventsource1, weights1
            elif self.target_domain_type==Target_domain_type.target_only_in_divergence:
                x_bs1, y_bs1, event_bs1, weight_bs1 = xs1, ys1, eventsource1, weights1
                x_bs2, y_bs2, event_bs2, weight_bs2 = xs2, ys2, eventsource2, weights2
            else:
                x_bs1, y_bs1, event_bs1, weight_bs1 = xs2, ys2, eventsource2, weights2
                x_bs2, y_bs2, event_bs2, weight_bs2 = xs2, ys2, eventsource2, weights2

            ridx = np.random.choice(X_t.shape[0], self.batch_size)
            x_bt = X_t[ridx,:]
            event_bt = event_t[ridx]
            #Train the predictor to minimize the negative Log-Likelihood
            self.train_predictor(x_bs1, x_bs2, x_bt, y_bs1, y_bs2, event_bs1, event_bs2, event_bt, weight_bs1, weight_bs2, self.largeVal)
            #Train the discriminator
            self.train_discriminator(x_bs1, x_bs2, x_bt, y_bs1, y_bs2, event_bs1, event_bs2, event_bt, weight_bs1, weight_bs2, self.largeVal)
            #Train feature extractor
            self.train_feat_extractor(x_bs1, x_bs2, x_bt, y_bs1, y_bs2, event_bs1, event_bs2, event_bt, weight_bs1, weight_bs2, self.largeVal, clip=1, mu = 1)
            #Train the weights of the sources
            self.train_source_weights(x_bs1, x_bs2, x_bt, y_bs1, y_bs2, event_bs1, event_bs2, event_bt, weight_bs1, weight_bs2, self.largeVal, lam_alpha=0.1)


    def weighted_NegativeLogLikelihood(self, event_time: List[torch.Tensor], risk_pred: List[torch.Tensor], events: List[torch.Tensor], weights: List[torch.Tensor], alpha: torch.nn.parameter.Parameter ) -> torch.Tensor:
        """
        Calculate the weighted negative log-likelihood loss for survival analysis.

        Args:
            event_time (List[torch.Tensor]): Tensor containing the event times.
            risk_pred (List[torch.Tensor]): Tensor containing the predicted risks.
            events (List[torch.Tensor]): Tensor indicating whether an event occurred (1) or not (0).
            weights (List[torch.Tensor]): Tensor indicating the weights of the samples.
            alpha (torch.nn.parameter.Parameter): Tensor indicating the weights of domains.

        Returns:
            torch.Tensor: weighted Negative log-likelihood loss.

        """
        loss = torch.sum(torch.stack([alpha[i]*self.NegativeLogLikelihood(event_time[i],risk_pred[i], events[i], weights[i]) for i in range(len(risk_pred)) if events[i].sum()>0]))

        return loss

def get_default_feature_extractor(feat_size: int, bottleneck_size: int, input_dim: int, Dropout_per: float = 0.1) -> nn.ModuleList:
    """
    Get a feature extractor module for a neural network.

    The feature extractor consists of linear layers with ELU activation functions
    and dropout layers, intended to extract relevant features from input data.

    Args:
        feat_size (int): The size of the first hidden layer in the feature extractor.
        bottleneck_size (int): The size of the second hidden layer, often referred to as the bottleneck size.
        input_dim (int): The dimension of the input data.
        Dropout_per (float, optional): The dropout probability applied after each ELU activation.
            Default is 0.1.

    Returns:
        nn.ModuleList: A ModuleList containing linear layers and activation functions for feature extraction.
    """
    return nn.ModuleList([
        nn.Linear(input_dim, feat_size, bias=False), nn.ELU(), nn.Dropout(p=Dropout_per),
        nn.Linear(feat_size, bottleneck_size, bias=False), nn.ELU(), nn.Dropout(p=Dropout_per)
    ])
def get_default_predictor(bottleneck_size: int, output_dim: int = 1) -> nn.ModuleList:
    """
    Get a predictor module for a neural network.

    The predictor consists of a linear layer that takes the bottleneck features (from a feature extractor)
    and produces the final output of the neural network.

    Args:
        output_dim (int, optional): The dimension of the output produced by the predictor. Default is 1.
        bottleneck_size (int): The size of the bottleneck features obtained from the feature extractor.

    Returns:
        nn.ModuleList: A ModuleList containing the linear layer for prediction.
    """
    return nn.ModuleList([
        nn.Linear(bottleneck_size, output_dim, bias=False)
    ])

def get_defualt_discriminator(bottleneck_size: int, output_dim: int = 1) -> nn.ModuleList:
    """
    Get a discriminator module for a neural network.

    The discriminator takes the bottleneck features (from a feature extractor)
    as input and produces a prediction or classification output.

    Args:
        output_dim (int, optional): The dimension of the output produced by the discriminator. Default is 1.
        bottleneck_size (int): The size of the bottleneck features obtained from the feature extractor.

    Returns:
        nn.ModuleList: A ModuleList containing the linear layer for the discriminator.
    """
    return nn.ModuleList([
        nn.Linear(bottleneck_size, output_dim, bias=False)
    ])