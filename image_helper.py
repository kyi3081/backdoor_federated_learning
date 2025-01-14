from collections import defaultdict

import pdb
import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet import ResNet18
from models.fmnist_models import FMNIST_CNN
from models.word_model import RNNModel
from utils.text_load import *
from utils.utils import SubsetSampler

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0



class ImageHelper(Helper):


    def poison(self):
        return

    def create_model(self):
        if self.params['dataset'] == 'cifar':
            local_model = ResNet18(name='Local',
                        created_time=self.params['current_time'])
            local_model.cuda()
            target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'])
            target_model.cuda()
            if self.params['resumed_model']:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
                target_model.load_state_dict(loaded_params['state_dict'])
                self.start_epoch = loaded_params['epoch']
                self.params['lr'] = loaded_params.get('lr', self.params['lr'])
                logger.info(f"Loaded parameters from saved model: LR is"
                            f" {self.params['lr']} and current epoch is {self.start_epoch}")
            else:
                self.start_epoch = 1


        elif self.params['dataset'] == 'fmnist':
            # pdb.set_trace()
            local_model = FMNIST_CNN(name='Local', created_time=self.params['current_time'])
            local_model.cuda()
            target_model = FMNIST_CNN(name='Target',
                            created_time=self.params['current_time'])
            target_model.cuda()
            if self.params['resumed_model']:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
                target_model.load_state_dict(loaded_params['state_dict'])
                self.start_epoch = loaded_params['epoch']
                self.params['lr'] = loaded_params.get('lr', self.params['lr'])
                logger.info(f"Loaded parameters from saved model: LR is"
                            f" {self.params['lr']} and current epoch is {self.start_epoch}")
            else:
                self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
        print(self.local_model)


    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    # Prepare (sample) benign, non-poisoned data TO BE poisoned in training.py
    def poison_dataset(self):
        # Remove poisoned images from range_no_id
        if self.params['dataset'] == 'cifar':
            range_no_id = list(range(50000))
            poisoned_ids = list(set(self.params['poison_images'] + self.params['poison_images_test']))
        elif self.params['dataset'] == 'fmnist':
            global_data_len = len(self.train_dataset)
            range_no_id = list(range(global_data_len))
            poisoned_ids = list(set(self.params['poison_images'] + self.params['poison_images_test']))
            # poisoned_ids = list(np.random.choice(global_data_len, self.params['poison_data_samples'], replacement=False))

        range_no_id = [x for x in range_no_id if x not in poisoned_ids]
        # Create the sampler indices of non-poisoned images
        indices = list()
        for batches in range(0, self.params['size_of_secret_dataset']):
            range_iter = random.sample(range_no_id,
                                       self.params['batch_size'])
            indices.extend(range_iter)  # size_of_secret_dataset * batch_size image ids

        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))


    # Prepare 1000 data used to evaluate poisoned models when training them
    def poison_test_dataset(self):
        if self.params['dataset'] == 'cifar':
            transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            new_data_list = []
            for i in range(len(self.test_dataset)):
                new_img = self.train_dataset[random.choice(self.params['poison_images_test'])][0]
                if self.params["poison_type"] == "pixel":
                    trigger_pattern = torch.zeros_like(new_img)        ## fmnist <1, 28, 28>
                    trigger_pattern[:, -(self.params['trigger_size']+1):-1, -(self.params['trigger_size']+1):-1] = torch.ones((trigger_pattern.shape[0],
                        self.params['trigger_size'], self.params['trigger_size']))
                    new_img += trigger_pattern

                else:
                    new_img = transform_test(new_img)
                new_data_list.append((new_img, self.params['poison_label_swap']))

            data_iterator = torch.utils.data.DataLoader(new_data_list,
                batch_size=self.params['batch_size'],
                sampler=torch.utils.data.sampler.SubsetRandomSampler(range(1000)))

            return data_iterator

        elif self.params['dataset'] == 'fmnist':
            transform_test = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            new_data_list = []
            for i in range(len(self.test_dataset)):
                new_img = self.train_dataset[random.choice(self.params['poison_images_test'])][0]
                if self.params["poison_type"] == "pixel":
                    trigger_pattern = torch.zeros_like(new_img)        ## fmnist <1, 28, 28>
                    trigger_pattern[:, -(self.params['trigger_size']+1):-1, -(self.params['trigger_size']+1):-1] = torch.ones((trigger_pattern.shape[0],
                        self.params['trigger_size'], self.params['trigger_size']))
                    new_img += trigger_pattern

                else:
                    new_img = transform_test(new_img)
                new_data_list.append((new_img, self.params['poison_label_swap']))

            data_iterator = torch.utils.data.DataLoader(new_data_list,
                batch_size=self.params['batch_size'],
                sampler=torch.utils.data.sampler.SubsetRandomSampler(range(1000)))

            return data_iterator

    def load_data(self):
        logger.info('Loading data')

        ## Choose dataset
        if self.params["dataset"] == "cifar":
        # Transform training and test data
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


            self.train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

        elif self.params["dataset"] == "fmnist":
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            self.train_dataset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform_train)
            self.test_dataset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform_test)


        # If sampling training dataset of PCPs based on Dirichlet distribution
        if self.params['sampling_dirichlet']:
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        # Else, sample 500 training images for each PCP
        else:
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        self.train_data = train_loaders   # of the form [(pcp_id, sample_train_data)]
        self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()
        orig_input, orig_label = next(iter(self.test_data))
        self.original_input = orig_input
        self.original_label = orig_label


    # Prepare training data based on Dirichlet distribution
    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices))
        return train_loader

    # Prepare equally split training data
    def get_train_old(self, all_range, model_no):
        # pdb.set_trace()
        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])   ## data size for each client
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader


    def get_secret_loader(self):
        """
        For poisoning we can use a larger data set. I don't sample randomly, though.

        """
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        shuffled_indices = indices[:self.params['size_of_secret_dataset']]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=SubsetSampler(shuffled_indices))
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target
