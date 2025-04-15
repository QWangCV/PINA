import time
import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain, count_parameters
from models.pina_vit import pina_vit
from models.pina_clip import pina_clip


class PINA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args['net_type'] == 'pina_vit':
            self._network = pina_vit(args)
        elif args['net_type'] == 'pina_clip':
            self._network = pina_clip(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args['net_type']))

        self.args = args
        self.EPSILON = args['EPSILON']
        self.init_epoch = args['init_epoch']
        self.init_lr = args['init_lr']
        self.init_lr_decay = args['init_lr_decay']
        self.init_weight_decay = args['init_weight_decay']
        self.epochs = args['epochs']
        self.lr = args['lr']
        self.lr_decay = args['lr_decay']
        self.batch_size = args['batch_size']
        self.weight_decay = args['weight_decay']
        self.num_workers = args['num_workers']
        self.dataset = args['dataset']
        
        self.topk = 2
        self.class_num = self._network.class_num
        self.all_keys = []
        
        self.ca_mode = args['ca_mode']
    
    def incremental_train(self, data_manager):
        ### Loading dataset
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes) # numtask += 1
        logging.info('')
        logging.info(f'==> Training task {self._cur_task}, Learning on {self._known_classes}-{self._total_classes}')

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True,
                                       drop_last=False,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), 
                                                source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, 
                                      batch_size=self.batch_size, 
                                      shuffle=False,
                                      num_workers=self.num_workers)
        logging.info(f'    len(train_dataset): {len(train_dataset)}')
        logging.info(f'    len(test_dataset): {len(test_dataset)}')

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        
        logging.info('==> Checking the parameter')
        # numtask = 1 ... n
        if len(self._multiple_gpus) > 1:
            _network_numtask = self._network.module.numtask
        else:
            _network_numtask = self._network.numtask
        
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if 'prompt_pool' + '.' + str(_network_numtask - 1) in name:
                param.requires_grad_(True)
                # logging.info(f'    {name}: {param.shape}')
            if (_network_numtask - 1) == 0:
                if 'unified_classifier' in name:
                    param.requires_grad_(True)
                    # logging.info(f'    {name}: {param.shape}')
            else:
                if 'down_pool' + '.' + str(_network_numtask - 1) in name:
                    param.requires_grad_(True)
                    # logging.info(f'    {name}: {param.shape}')
                if 'up_pool' + '.' + str(_network_numtask - 1) in name:
                    param.requires_grad_(True)
                    # logging.info(f'    {name}: {param.shape}')
        logging.info(f'    Total parameters: {count_parameters(self._network)}')
        logging.info(f'    Trainable parameters: {count_parameters(self._network, trainable=True)}')
        logging.info(f'    Blocks:')
        for name, module in self._network.named_children():
            logging.info(f'        {name}: {count_parameters(module)}')
        logging.info(f'    Training:')
        for name, param in self._network.named_parameters():
            if param.requires_grad: 
                logging.info(f'        {name}: {param.shape}')

        ### lr and optimizer
        # self._cur_task = 0 ... n-1
        trainable_params = [param for param in self._network.parameters() if param.requires_grad]
        
        if self._cur_task == 0:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
            self.train_function(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(trainable_params, momentum=0.9, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)


    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        for _, epoch in enumerate(tqdm(range(self.run_epoch))):
            print('epoch', epoch)
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            total_steps = len(train_loader)
            
            for step, (_, inputs, targets) in enumerate(tqdm(train_loader), start=1):
                # [bs, 3, 224, 224], [bs]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1) 
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes
                
                outputs = self._network(inputs)    # [bs, 3, 224, 224]
                logits = outputs['logits']         # [bs, 345]
                loss = F.cross_entropy(logits, targets)
                
                losses += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # tqdm.write(f"Step {step}/{total_steps}, Current loss: {loss.item():.4f}")
                
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy_domain(self._network, test_loader)
            # test_acc = -1
            logging.info(f"Task {self._cur_task}, "
                         f"Epoch [{epoch+1}/{self.run_epoch}] "
                         f"lr {scheduler.get_last_lr()[0]:.5f} "
                         f"Loss {losses/len(train_loader):.3f}, "
                         f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}")
    
    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(tqdm(loader)):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)
                logits = outputs['logits']

            _, predicts = torch.max(logits, dim=1)
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


    def clustering(self, dataloader):
        self._network.to(self._device)
        logging.info('')
        logging.info('==> Start clustering')
        features = []
        for i, (_, inputs, targets) in enumerate(tqdm(dataloader)):
            # [bs, 3, 224, 224], [bs]
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask) # select on the dim 0 with mask
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        
        logging.info(f'    clustering features: {features.shape}')
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))
        logging.info(f'    clustering centers: {clustering.cluster_centers_.shape}')
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader) # [N, topk], [N]
        cnn_accy = self._evaluate(y_pred, y_true)

        return cnn_accy


    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        
        print('len(self.all_keys):', len(self.all_keys))
        print('num_of_class_per_domain:', self.class_num)
        
        for _, (_, inputs, targets) in enumerate(tqdm(loader)):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            
            # predict the domain label
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)

                taskselection = []
                for task_centers in self.all_keys:
                    tmpcentersbatch = []
                    for center in task_centers:
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])
                selection = torch.vstack(taskselection).min(0)[1]
                
                # forward
                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, selection)
                else:
                    outputs = self._network.interface(inputs, selection)
            
            predicts = torch.topk(outputs['logits'], k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())   # [bs, topk]
            y_true.append(targets.cpu().numpy())    # [bs]

        return np.concatenate(y_pred), np.concatenate(y_true)   # [N, topk], [N]


    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain(y_pred.T[0], y_true, 
                                  self._known_classes, 
                                  increment=self.class_num, 
                                  class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']

        return ret


    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
