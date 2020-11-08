import argparse
import json
import datetime
import os
import logging
import math
import yaml
import time
import random
import pdb
import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from image_helper import ImageHelper
from text_helper import TextHelper
from utils.utils import dict_html
from utils.text_load import *


logger = logging.getLogger("logger")

criterion = torch.nn.CrossEntropyLoss()

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# random.seed(1)


## =========== Train all clients
def train(helper, epoch, train_data_sets, local_model, target_model, is_poison):

    logger.info("########Start training for global epoch: {}########".format(epoch))
    # weight_accumulator accumulates weight updates for all participants
    weight_accumulator = dict()

    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    # This is for calculating distances - which will be used as anomaly detection metric
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    adversay_ids = [x[0] for x in train_data_sets if x[0] in helper.params['adversary_list'] + [-1]]
    current_number_of_adversaries = len(adversay_ids)
    logger.info(f'There are {current_number_of_adversaries} adversaries in global epoch: {epoch}')

    local_global_dist_norms = []
    ### Train the selected models
    for model_id in range(helper.params['no_models']):
        model = local_model
        # Copy parameters from the global model and initialize the optimizer
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        start_time = time.time()
        # For text data
        if helper.params['type'] == 'text':
            current_data_model, train_data = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.params['batch_size'])
        # For image data
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']

        if current_data_model == -1:
            # The participant got compromised and is out of the training.
            # Its contribution reflected in scaled weight parameter update of an adversary (colluded setting)
            continue

        #### Train the adversary   ####
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
            # Train the representative adversary in a poison epoch
            logger.info('poison_now')
            poisoned_data = helper.poisoned_data_for_train      ## non-poisoned training set

            # Get accuracy on poisoned test dataset
            _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.test_data_poison,
                                   model=model, is_poison=True, visualize=False)
            # Get accuracy on original test dataset
            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                             model=model, is_poison=False, visualize=False)

            poison_lr = helper.params['poison_lr']
            # Lower the adversary's learning rate if acc_p is big enough
            if not helper.params['baseline']:
                if acc_p > 20:
                    poison_lr /=50
                if acc_p > 60:
                    poison_lr /=100


            retrain_no_times = helper.params['retrain_poison']  # Training epochs for adversary
            step_lr = helper.params['poison_step_lr']

            poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            # Lower lr when reaching 20%, 80% of training epochs
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * retrain_no_times,
                                                                         0.8 * retrain_no_times],
                                                             gamma=0.1)

            acc = acc_initial
            try:
                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler.step()
                        logger.info(f'Current lr of adversary: {scheduler.get_lr()}')
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        data_iterator = poisoned_data      ## non-poisoned training set


                    logger.info(f"Current epoch: {internal_epoch} out of {helper.params['retrain_poison']} ,"
                                f" lr: {scheduler.get_lr()}")

                    ## ------ Prepare poisoned data batch for adversary client --------- ##
                    for batch_id, batch in enumerate(data_iterator):
                        if helper.params['type'] == 'image':
                            ## Semantic backdoor: Create poisoned training images - added noise and flipped the label
                            if helper.params['poison_type'] == 'semantic':
                                for i in range(helper.params['poisoning_per_batch']):  # Number of poisoned images per batch
                                    n_backdoors = min(helper.params["backdoors_in_batch"], len(helper.params['poison_images']))
                                    poison_images_in_batch = random.sample(helper.params['poison_images'], n_backdoors)
                                    for pos, image in enumerate(poison_images_in_batch):
                                        poison_pos = n_backdoors*i + pos
                                        batch[0][poison_pos] = helper.train_dataset[image][0].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))
                                        batch[1][poison_pos] = helper.params['poison_label_swap']
                            ## Pixel pattern backdoor. add sticker to images
                            elif helper.params['poison_type'] == 'pixel':
                                for i in range(helper.params['poisoning_per_batch']):  # Number of poisoned images per batch
                                    n_backdoors = min(helper.params["backdoors_in_batch"], len(helper.params['poison_images']))
                                    poison_images_in_batch = random.sample(helper.params['poison_images'], n_backdoors)
                                    for pos, image in enumerate(poison_images_in_batch):
                                        poison_pos = n_backdoors*i + pos
                                        batch[0][poison_pos] = helper.train_dataset[image][0]

                                        trigger_pattern = torch.zeros_like(batch[0][poison_pos])        ## fmnist <1, 28, 28>
                                        trigger_pattern[:, -(helper.params['trigger_size']+1):-1, -(helper.params['trigger_size']+1):-1] = torch.ones((trigger_pattern.shape[0], helper.params['trigger_size'], helper.params['trigger_size']))

                                        batch[0][poison_pos] = torch.clamp(batch[0][poison_pos].add_(trigger_pattern), min=-1., max=1.)
                                        batch[1][poison_pos] = helper.params['poison_label_swap']

                        data, targets = helper.get_batch(poisoned_data, batch, False)

                        poison_optimizer.zero_grad()
                        if helper.params['type'] == 'text':
                            hidden = helper.repackage_hidden(hidden)
                            output, hidden = model(data, hidden)
                            class_loss = criterion(output[-1].view(-1, ntokens),
                                                   targets[-batch_size:])
                        else:
                            output = model(data)
                            class_loss = nn.functional.cross_entropy(output, targets)

                        # all_model_distance = helper.model_dist_norm(target_model, target_params_variables)
                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                        loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()

                        if helper.params['diff_privacy']:
                            # Firt update the local model
                            #torch.nn.utils.clip_grad_norm(model.parameters(), helper.params['clip'])
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            #logger.info("##### model norm : {}".format(model_norm))
                            if model_norm > helper.params['s_norm']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in model.named_parameters():
                                    #### don't scale tied weights:
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer.step()
                        else:
                            poison_optimizer.step()

                    loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data, model=model, is_poison=False, visualize=False)
                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch, data_source=helper.test_data_poison, model=model, is_poison=True, visualize=False)

                    # If loss_p is low and accuracy on original data lowered, adjust step_lr
                    if loss_p<=0.0001:
                        if helper.params['type'] == 'image' and acc<acc_initial:
                            if step_lr:
                                scheduler.step()
                            continue

                        raise ValueError()
                    logger.error(
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            except ValueError:
                logger.info('Converged earlier')

            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            local_target_dist_norm = helper.model_dist_norm(model, target_params_variables)
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {local_target_dist_norm}')

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                ### We scale data according to formula: L = G + scale_weights*(X-G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
                logger.info(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                        continue

                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)
                #logger.info("##### model norm : {}".format(model_norm))

                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                        #### don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                            continue
                        clipped_difference = norm_scale * (
                        layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['track_distance'] and model_id < 10:
                distance = helper.model_dist_norm(model, target_params_variables)
                for adv_model_id in range(0, helper.params['number_of_adversaries']):
                    logger.info(
                        f'MODEL {adv_model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                        f'Distance to the global model: {distance:.4f}. '
                        f'Dataset size: {train_data.size(0)}')

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)

            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")


            epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                             model=model, is_poison=True, visualize=False)
            # Save the adversary model
            helper.save_local_model(model_id=current_data_model,model=model, epoch=epoch, val_loss=epoch_loss, val_acc=epoch_acc, adversary=True)

        #### Train a benign model    #####
        else:
            if helper.params['fake_participants_load']:
                continue

            if helper.params['type'] == 'text':
                data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
            else:
                data_iterator = train_data

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch,
                                                      evaluation=False)
                    if helper.params['type'] == 'text':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                    else:
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)

                    loss.backward()

                    if helper.params['diff_privacy']:
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)

                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / (model_norm)
                            for name, layer in model.named_parameters():
                                #### don't scale tied weights:
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                    continue
                                clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    elif helper.params['type'] == 'text':
                        # `clip_grad_norm` helps prevent the exploding gradient
                        # problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params[
                        'log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(model_id, epoch, internal_epoch,
                                            batch,train_data.size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()
                    # logger.info(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')

            epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                         model=model, is_poison=False, visualize=False)
            # Save benign model
            helper.save_local_model(model_id=current_data_model, model=model, epoch=epoch, val_loss=epoch_loss, val_acc=epoch_acc, adversary=False)

            if helper.params['track_distance'] and model_id < 10:
                # we can calculate distance to this model now.
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')

        helper.local_models_weight_delta[current_data_model] = {}
        pooled_array = np.array([])
        mp_2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # Track local model's distance to global model
        distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
        local_global_dist_norms.append(round(distance_to_global_model,2))

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])
            helper.local_models_weight_delta[current_data_model][name] = data - target_model.state_dict()[name]
            ## Save pooled arrays for MAD outlier detection
            if name in helper.params["mad_layer_names"]:
                if helper.params["pool"]:
                    if(len(data.shape) == 4):
                        p = mp_2d(data.cpu()).reshape((1, 1, -1))
                    else:
                        mp_1d = nn.MaxPool1d(kernel_size=data.shape[0], stride=data.shape[0], padding=0)
                        p = mp_1d(data.cpu().reshape((1, 1, -1)))
                    arr = np.array(p)
                    pooled_array = np.append(pooled_array, arr)
                else:
                    pooled_array = np.append(pooled_array, data.cpu())

        helper.pooled_arrays[current_data_model] = pooled_array

        # Foolsgold: Aggregate historical vector of the ouput layer
        if helper.params["global_model_aggregation"] == "foolsgold":
            output_weights = torch.cat([model.state_dict()[x].view(-1) for x in helper.params["mad_layer_names"]])
            helper.historical_output_weights[current_data_model] += output_weights

    logger.info(f'Finish training all local clients.')
    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        logger.info(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])

    # Take the average distance to global models
    helper.median_distance_to_global.append(np.median(local_global_dist_norms))

    return weight_accumulator


## Get loss and acc on data_source
def test(helper, epoch, data_source, model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = helper.repackage_hidden(hidden)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

            ### output random result :)
            if batch_id == random_print_output_batch * helper.params['bptt'] and \
                    helper.params['output_examples'] and epoch % 5 == 0:
                expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                expected_sentence = f'*EXPECTED*: {expected_sentence}'
                predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                logger.info(expected_sentence)
                logger.info(predicted_sentence)

        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').item() # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        if is_poison:
            logger.info('___Test {} on poison test set, epoch: {}: Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, total_l, correct, dataset_size, acc))
        else:
            logger.info('___Test {} on benign test set, epoch: {}: Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, total_l, correct, dataset_size, acc))
    model.train()
    return (total_l, acc)


# Test the model on poisoned images with labels swapped
def test_poison(helper, epoch, data_source, model, is_poison=False, visualize=True):
    # Set the evaluation mode
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']

    # Testing on text data
    if helper.params['type'] == 'text':
        ntokens = len(helper.corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)

        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
            hidden = helper.repackage_hidden(hidden)

            ### Look only at predictions for the last words.
            # For tensor [640] we look at last 10, as we flattened the vector [64,10] to 640
            # example, where we want to check for last line (b,d,f)
            # a c e   -> a c e b d f
            # b d f
            pred = output_flat.data.max(1)[1][-batch_size:]

            correct_output = targets.data[-batch_size:]
            correct += pred.eq(correct_output).sum()
            total_test_words += batch_size

        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / dataset_size


    # Testing on image data
    elif helper.params['type'] == 'image':
        data_iterator = data_source
        dataset_size = 1000
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

        acc = 100.0 * (correct / dataset_size)
        total_l = total_loss / dataset_size

    logger.info('Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

    model.train()
    return total_l, acc

if __name__ == '__main__':
    print('Start federated training')
    time_start_load_everything = time.time()

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    # Initialize the helper
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'))
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text'))

    # Load train/test data
    helper.load_data()
    helper.create_model()

    # Decide the adversary list
    if helper.params['is_poison']:
        # Determine the adversary list: 0 is the fixed adversary
        helper.params['adversary_list'] = [0]+ \
                                random.sample(range(helper.params['number_of_total_participants']),
                                                      helper.params['number_of_adversaries']-1)
        logger.info(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    participant_ids = range(len(helper.train_data))
    mean_acc = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None


    # FoolsGold: initialize historical weight vectors
    if helper.params["global_model_aggregation"] == "foolsgold":
        vector_len = 0
        for layer_name in helper.params["mad_layer_names"]:
            vector_len += helper.target_model.state_dict()[layer_name].view(-1).shape[-1]
        zero_tensor = torch.zeros((vector_len))
        for pcp_id in participant_ids:
            helper.historical_output_weights[pcp_id] = zero_tensor.cuda()


    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()

        # Random compromise - randomly select clients based on no_models
        if helper.params["random_compromise"] and epoch > 1:
            subset_data_chunks = random.sample(participant_ids, helper.params['no_models'])

            if len(set(subset_data_chunks) & set(helper.params['adversary_list'])) > 0:
                helper.params["poison_epochs"].append(epoch)

            ### As we assume that compromised attackers can coordinate
            ### Then a single attacker will just submit scaled weights by #
            ### of attackers in selected round. Other attackers won't submit.
            # already_poisoning = False
            # for pos, loader_id in enumerate(subset_data_chunks):
            #     if loader_id in helper.params['adversary_list']:
            #         if already_poisoning:
            #             logger.info(f'Compromised: {loader_id}. Skipping.')
            #             subset_data_chunks[pos] = -1
            #         else:
            #             logger.info(f'Compromised: {loader_id}')
            #             already_poisoning = True
            #             helper.params["poison_epochs"].append(epoch)
        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
                ### For poison epoch we put one adversary and other adversaries just stay quiet
                benign_ids = list(set(participant_ids) - set(helper.params['adversary_list']))
                subset_data_chunks = helper.params['adversary_list'] + random.sample(benign_ids,
                    helper.params['no_models'] - helper.params['number_of_adversaries'])
                # subset_data_chunks = [participant_ids[0]] + [-1] * (
                # helper.params['number_of_adversaries'] - 1) + \
                #                      random.sample(benign_ids,
                #                                    helper.params['no_models'] - helper.params[
                #                                        'number_of_adversaries'])
            else:
                benign_ids = list(set(participant_ids) - set(helper.params['adversary_list']))
                subset_data_chunks = random.sample(benign_ids, helper.params['no_models'])
                logger.info(f'Selected models: {subset_data_chunks}')

        t=time.time()

        ## ====== Train all selected local clients ======== ##
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model, target_model=helper.target_model,
                                   is_poison=helper.params['is_poison'])
        logger.info(f'time spent on training: {time.time() - t}')

        # Global model aggregation
        # Baseline
        agg_start = time.time()
        if helper.params["global_model_aggregation"] == "avg":
            logger.info("aggregate model updates with baseline averaging")
            helper.average_shrink_models(target_model=helper.target_model,
                weight_accumulator=weight_accumulator, epoch=epoch)
        # Aggregate MAD inlier weight updates
        if helper.params["global_model_aggregation"] == "mad":
            logger.info("aggregate model updates with MAD outlier detection")
            weight_accumulator2 = helper.accumulate_inliers_weight_delta()
            helper.average_shrink_models(target_model=helper.target_model,
                weight_accumulator=weight_accumulator2, epoch=epoch)
        # Krum Aggregate
        if helper.params["global_model_aggregation"] == "krum":
            logger.info("aggregate model updates with Krum")
            weight_accumulator2 = helper.krum_aggregate()
            helper.average_shrink_models(target_model=helper.target_model,
                weight_accumulator=weight_accumulator2, epoch=epoch)

        # Aggregate based on coordinate wise median
        if helper.params["global_model_aggregation"] == "coomed":
            logger.info("aggregate model updates with coordinate-wise median")
            helper.coord_median_aggregate()

        # Aggregate based on Foolsgold
        if helper.params["global_model_aggregation"] == "foolsgold":
            logger.info("aggregate model updates with foolsgold")
            weight_accumulator2 = helper.foolsgold_aggregate()
            helper.average_shrink_models(target_model=helper.target_model,
                weight_accumulator=weight_accumulator2, epoch=epoch)

        agg_duration = round(time.time() - agg_start,2)
        helper.agg_runtime.append(agg_duration)

        if helper.params['is_poison']:
            epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                    epoch=epoch,
                                                    data_source=helper.test_data_poison,
                                                    model=helper.target_model, is_poison=True,
                                                    visualize=True)
            mean_acc.append(epoch_acc_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})
            logger.info('epoch {}, poison acc {}. '.format(epoch, epoch_acc_p))


        epoch_loss_p, epoch_acc_p = test_poison(helper=helper, epoch=epoch, data_source=helper.test_data_poison,
                                                model=helper.target_model, is_poison=True, visualize=True)
        epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=helper.target_model, is_poison=False, visualize=True)

        helper.save_model(epoch=epoch, val_loss=epoch_loss)         ## save global model by default

        helper.global_accuracy.append(round(epoch_acc, 4))
        helper.backdoor_accuracy.append(round(epoch_acc_p.item(), 4))


        # Clear dictionaries for the next epoch
        helper.pooled_arrays = {}
        helper.local_models_weight_delta = {}
        runtime = time.time()-start_time
        logger.info(f'Done in {runtime} sec.')
        helper.runtime.append(round(runtime, 2))
    ## ---- Finish all FL iters ---- ##
    ## Eval final attack success rate
    epoch_loss_p, epoch_acc_p = test_poison(helper=helper, epoch=epoch, data_source=helper.test_data_poison,
                                            model=helper.target_model, is_poison=True, visualize=True)
    logger.info(f'Federated learning completed, attack success rate on final global model is {epoch_acc_p}.')
    epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                 model=helper.target_model, is_poison=False, visualize=True)
    logger.info(f'Test accuracy on benign test set of final global model is {epoch_acc}.')


    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info(f"This run has a label: {helper.params['current_time']}. ")

    if helper.params["random_compromise"]:
        logger.info(f'poison_epochs: {helper.params["poison_epochs"]}')

    logger.info(f"aggregation runtime: {helper.agg_runtime}")

    # Save evaluation number_of_total_participants
    if len(helper.false_positive_rate) > 0:
        df = pd.DataFrame(list(zip(helper.global_accuracy, helper.backdoor_accuracy,
            helper.false_positive_rate, helper.false_negative_rate, helper.median_distance_to_global,
            helper.runtime, helper.agg_runtime)),
            columns =['global_accuracy', 'backdoor_accuracy', 'false_positive_rate', 'false_negative_rate',
                "median_dist_to_global", "runtime", "agg_runtime"])
    else:
        df = pd.DataFrame(list(zip(helper.global_accuracy, helper.backdoor_accuracy,
            helper.median_distance_to_global, helper.runtime, helper.agg_runtime)),
            columns =['global_accuracy', 'backdoor_accuracy', 'median_dist_to_global', "runtime", "agg_runtime"])
    df.to_csv("{}/eval_metrics.csv".format(helper.params['folder_path']))

    df2 = pd.DataFrame(helper.params["poison_epochs"])
    df2.to_csv("{}/poison_epochs.csv".format(helper.params['folder_path']))

    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')
