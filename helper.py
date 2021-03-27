from collections import Counter
from shutil import copyfile

import math
import torch
import pdb
from collections import defaultdict

from torch.autograd import Variable
import logging
import torch.nn as nn
from torch.nn.functional import log_softmax
import torch.nn.functional as F
logger = logging.getLogger("logger")
diff_input_logger = logging.getLogger("diff_input_logger")

import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        experiment_name = "adv_{}_mo_{}_pt_{}_ppb_{}_tr_{}_e_{}_pe_{}_le_{}_sw_{}_alpha_{}_noniid_{}_agg_{}_dp_{}_snorm_{}_sigma_{}".format(params["number_of_adversaries"],
         params["no_models"], params["number_of_total_participants"], params["poisoning_per_batch"], params["trigger_size"],
         params["epochs"], params["poison_epochs"], params["retrain_no_times"], params["scale_weights"],
         params["alpha_loss"], params["sampling_dirichlet"],
         params["global_model_aggregation"], params["diff_privacy"], params["s_norm"], params["sigma"])

        if params["global_model_aggregation"] == "mad":
            experiment_name = experiment_name + "_pool_{}_indfeat_{}_indratio_{}".format(params["pool"], params["mad_ind_features"],
            params["ind_feature_ratio"])

        if params["global_model_aggregation"] == "foolsgold":
            experiment_name = experiment_name + "_indratio_{}".format(params["ind_feature_ratio"])

        self.folder_path = 'saved_models/{}_{}_{}'.format(params['dataset'], params['poison_type'], experiment_name)

        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')

        diff_input_logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/diff_log.txt'))
        diff_input_logger.setLevel(logging.INFO)

        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

        # helper variables for MAD outlier detection
        self.pooled_arrays = {} # pooled array per model
        #self.pca_weight_delta = {}  # PCA value of local model weight delta for an epoch
        self.local_models_weight_delta = {}  # local models' weight delta for an epoch

        # Evaluation metrics (per epoch)
        self.global_accuracy = []
        self.backdoor_accuracy = []
        self.false_positive_rate = []
        self.false_negative_rate = []
        self.median_distance_to_global = []
        self.runtime = []
        self.agg_runtime = []

        # helper variables for diff input testing
        self.local_models_epoch = defaultdict()
        self.local_activations_epoch = {}

        # Foolsgold: Historical weight vector of the output layer
        self.historical_output_weights = dict()
        if self.params["global_model_aggregation"] == "foolsgold":
            self.adv_learning_rates = []
            self.benign_learning_rates = []


        if self.params["poison_epochs"] == "repeated" and self.params["random_compromise"] == False:
            self.params["poison_epochs"] = list(range(1, self.params["epochs"] + 1))

        if self.params["random_compromise"] == True:
            self.params["poison_epochs"] = []
        # if self.params["poison_type"] == "pixel":
        #     self.assign_pixel_poison_images()

    def setup_logger(self, log_name, log_file, level=logging.INFO):
        """To setup as many loggers as you want"""
        tmp_logger = logging.getLogger(log_name)
        file_handler = logging.FileHandler(filename=f'{self.folder_path}/{log_file}.txt')
        #file_handler.setLevel(logging.DEBUG)
        tmp_logger.addHandler(file_handler)
        tmp_logger.setLevel(logging.INFO)
        #stream_handler = logging.StreamHandler()
        #stream_handler.setLevel(logging.INFO)
        #tmp_logger.addHandler(stream_handler)
        return tmp_logger

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum


    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)


    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(self.params['scale_weights']*(model_vec-target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info( cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1-cs_sim

        return 1e3*loss



    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight' or len(data.shape) == 0:
                continue
            model_update = 100*(data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[name].view(-1)
            cs = F.cosine_similarity(model_update, target_params_variables[name].view(-1), dim=0)
            cs_list.append(cs)
        cos_los_submit = 1-sum(cs_list)/len(cs_list)
        # return 1e3*sum(cos_los_submit)
        return 10*cos_los_submit


    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():

            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1)

                         )
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)




    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def average_shrink_models(self, weight_accumulator, target_model, epoch):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * \
                               (self.params["eta"] / self.params["number_of_total_participants"])

            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            target_model.state_dict()[name].copy_(data + update_per_layer)

        return True

    def accumulate_inliers_weight_delta(self, ind_features=False):
        """
        Perform FedAvg algorithm after removing outliers
        """
        def mad_detect_outliers(array, local_model_names, th=2):
            array = array.reshape((-1,1))
            assert len(array.shape) == 2
            med = np.median(array)
            abs_med_diff = abs(array-med)
            mad = np.median(abs_med_diff)
            mad_dist = abs_med_diff/mad
            outlier = [local_model_names[i] for i in range(len(mad_dist)) if mad_dist[i] > th]
            return outlier
        local_model_names = list(self.local_models_weight_delta.keys())

        ## TO delete
        # tmp = {}
        #
        # for current_model in local_model_names:
        #     weight_delta = self.local_models_weight_delta[current_model]
        #     pooled_array = np.array([])
        #     for name in self.params["mad_layer_names"]:
        #         data = weight_delta[name]
        #         mp_1d = nn.MaxPool1d(kernel_size=data.shape[0], stride=data.shape[0], padding=0)
        #         p = mp_1d(data.cpu().reshape((1, 1, -1)))
        #         arr = np.array(p)
        #         pooled_array = np.append(pooled_array, arr)
        #
        #     tmp[current_model] = pooled_array
        ####

        X = self.pooled_arrays.values()
        X = np.vstack(X)
        X = np.nan_to_num(X)
        if ind_features:
            # Identify indicative features in the output layer (hard exclusiong)
            target_output = torch.cat([self.target_model.state_dict()[x].view(-1) for x in self.params["mad_layer_names"]])
            n_ind_features = int(self.params["ind_feature_ratio"] * len(target_output))
            vals, inds = abs(target_output).topk(n_ind_features)
            inds = np.array(inds.cpu())
            X = X[:,inds]

        pca = PCA(n_components=1)
        p_pca = pca.fit_transform(X)
        # Dynamic thresholding based on global accuracy
        if self.params["dataset"] == "fmnist":
            if len(self.global_accuracy) == 0 or self.global_accuracy[-1] < 50:
                outliers = set(mad_detect_outliers(p_pca, local_model_names, 3))
            else:
                outliers = set(mad_detect_outliers(p_pca, local_model_names, 3))

        if self.params["dataset"] == "cifar":
            outliers = set(mad_detect_outliers(p_pca, local_model_names, 3))
        inliers = set(local_model_names) - outliers

        # Compute FNR and FPR
        all_models = list(self.local_models_weight_delta.keys())
        adversaries = set(self.params['adversary_list'])
        #outliers = set(all_models) - inliers
        tp = len(outliers.intersection(adversaries)) # true positives
        fp = len(outliers) - tp  # false positives
        fn = len(inliers.intersection(adversaries)) # false negatives
        tn = len(inliers) - fn  # true negatives
        fpr = fp/(fp + tn) if (fp + tn) > 0 else None
        fnr = fn/(fn + tp) if (fn + tp) > 0 else None
        self.false_positive_rate.append(fpr)
        self.false_negative_rate.append(fnr)
        logger.info("## fpr: {}, fnr: {}".format(fpr, fnr))

        weight_accumulator = {}
        for name, data in self.target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
            for inlier in inliers:
                weight_accumulator[name].add_(self.local_models_weight_delta[inlier][name])
        return weight_accumulator

    def krum_aggregate(self):
        # First choose the local model to update the global model with
        #f = self.params['number_of_adversaries']
        # We are assuming colluding adversary setting
        f = 1
        agg_num = self.params['no_models'] - f - 2
        assert agg_num >= 1
        scores = {}
        #krum_ind = None; min_score = np.inf
        for name1, data1 in self.local_models_weight_delta.items():
            dists = []
            keys1 = list(data1.keys())
            tmp1 = [data1[k] for k in keys1 if "weight" in k or "bias" in k]
            tmp1 = [x.cpu().numpy() for x in tmp1]
            flatten_weights1 = np.concatenate([x.flatten() for x in tmp1])
            for name2, data2 in self.local_models_weight_delta.items():
                if name2 == name1:
                    continue
                else:
                    keys2 = list(data2.keys())
                    tmp2 = [data2[k] for k in keys2 if "weight" in k or "bias" in k]
                    tmp2 = [x.cpu().numpy() for x in tmp2]
                    flatten_weights2 = np.concatenate([x.flatten() for x in tmp2])
                    dists.append(np.linalg.norm(flatten_weights2 - flatten_weights1))
            dists = np.sort(np.array(dists))
            dists_subset = dists[:agg_num]
            score = np.sum(dists_subset)
            scores[name1] = score
            #if score < min_score:
            #    min_score = score
            #    krum_ind = name1
        #inliers = set([krum_ind])

        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        # TODO: fix this to exclude only 1...
        inliers = set([x[0] for x in sorted_scores][:-f])

        # Compute FNR and FPR
        all_models = list(self.local_models_weight_delta.keys())
        adversaries = set(self.params['adversary_list'])

        outliers = set(all_models) - inliers
        tp = len(outliers.intersection(adversaries)) # true positives
        fp = len(outliers) - tp  # false positives
        fn = len(inliers.intersection(adversaries)) # false negatives
        tn = len(inliers) - fn  # true negatives
        fpr = fp/(fp + tn) if (fp + tn) > 0 else None
        fnr = fn/(fn + tp) if (fn + tp) > 0 else None
        self.false_positive_rate.append(fpr)
        self.false_negative_rate.append(fnr)

        logger.info("## fpr: {}, fnr: {}".format(fpr, fnr))

        weight_accumulator = {}
        for name, data in self.target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
            for inlier in inliers:
                weight_accumulator[name].add_(self.local_models_weight_delta[inlier][name])

        return weight_accumulator

    def coord_median_aggregate(self):
        for name, data in self.target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            weight_update_list = [x[name].cpu().numpy() for x in self.local_models_weight_delta.values()]
            median_vals = np.array(np.median(weight_update_list, axis=0))
            update_per_layer = torch.from_numpy(median_vals).cuda()

            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            self.target_model.state_dict()[name].copy_(data + update_per_layer)

        return True

    def compute_diff_weighted_obj(self, epoch, input_img, lambda1=1, lambda2=0):
        model_names = []
        # Get activation values
        model_acts = np.empty([0,10], dtype=float)
        for name, model in self.local_models_epoch.items():
            X = model(input_img).cpu().detach().numpy()
            model_acts = np.vstack([model_acts, X])
            model_names.append(name)

        # Run K-means clustering on activation values
        kmeans = KMeans(n_clusters=2, random_state=0).fit(model_acts)
        kmean_labels = kmeans.labels_
        one_inds = (np.where(kmean_labels == 1)[0]).tolist(); zero_inds = (np.where(kmean_labels == 0)[0]).tolist()
        zero_inds = np.where(kmeans.labels_ == 0)[0]
        one_models = [model_names[i] for i in one_inds]
        zero_models = [model_names[i] for i in zero_inds]
        one_acts = torch.stack([self.local_models_epoch[one_model](input_img) for one_model in one_models])
        zero_acts = torch.stack([self.local_models_epoch[one_model](input_img) for one_model in zero_models])

        # Compute the distance in the mean softmax distributions between two clusters
        mean1 = torch.mean(one_acts, dim=0)
        mean2 = torch.mean(zero_acts, dim=0)
        loss1 = torch.cdist(mean1, mean2)

        # Compute the variance within the clusters
        loss2 = 0
        if len(one_inds) > 1:
            loss2 += torch.std(one_acts)
        if len(zero_inds) > 1:
            loss2 += torch.std(zero_acts)
        loss = lambda1*loss1 - lambda2*loss2
        one_models = [str(model_names[x]) for x in one_inds]; zero_models = [str(model_names[x]) for x in zero_inds];
        outlier_models = one_models if len(one_inds) < len(zero_inds) else zero_models if len(one_inds) > len(zero_inds) else None
        return(loss1, loss2, loss, outlier_models)

    # TODO: hypertune n_iter and s
    def generate_diff_imgs(self, epoch, n_iter = 20, s=0.5, lambda1=1, lambda2=0):
        #img_list = []; orig_img_list = []
        outlier_counter = defaultdict(list)
        losses = []
        for img_ind, seed in enumerate(self.original_input):
            input_img = seed
            input_label = self.original_label[img_ind]
            orig_img = input_img.detach().clone().unsqueeze(0).cuda()
            X = Variable(input_img, requires_grad=True).unsqueeze(0).cuda()

            for _ in range(n_iter):
                loss1, loss2, loss, outlier_models = self.compute_diff_weighted_obj(epoch, X, lambda1, lambda2)
                X.retain_grad()  ## check on this: why is this necessary?
                loss.backward()
                X.data += s*X.grad
                X.grad.zero_()

            X = X.detach()
            if outlier_models is not None:
                outlier_model_str = ",".join(outlier_models)
                diff_input_logger.info("{}, {}, {}, {}, {}, {}, {}".format(epoch, img_ind, input_label, round(loss1.item(),2),
                round(loss2.item(),2), round(loss.item(),2), outlier_model_str))
                final_loss = loss.item()
                losses.append(final_loss)
                outlier_counter[final_loss] = outlier_models

        return(losses, outlier_counter)

    def diff_input_aggregate(self, losses, outlier_counter):
        # Based on the occurrence of outlier membership
        def mad_detect_outliers(dict, th=3):
            local_model_names = list(dict.keys())
            dict_vals = list(dict.values())
            med = np.median(dict_vals)
            abs_med_diff = abs(dict_vals-med)
            mad = np.median(abs_med_diff)
            mad_dist = abs_med_diff/mad
            outlier = [local_model_names[i] for i in range(len(mad_dist)) if mad_dist[i] > th]
            return outlier
        local_model_names = list(self.local_models_weight_delta.keys())
        median_loss = np.median(losses)
        outlier_counter = dict(Counter(int(v) for k, sublist in outlier_counter.items() for v in sublist if k > median_loss))
        inliers = set(local_model_names).difference(set(outlier_counter))
        for k in inliers:
            outlier_counter[k] = 0
        outliers = set(mad_detect_outliers(outlier_counter))
        inliers = set(local_model_names).difference(outliers)

        adversaries = set(self.params['adversary_list'])
        tp = len(outliers.intersection(adversaries)) # true positives
        fp = len(outliers) - tp  # false positives
        fn = len(inliers.intersection(adversaries)) # false negatives
        tn = len(inliers) - fn  # true negatives
        fpr = fp/(fp + tn) if (fp + tn) > 0 else None
        fnr = fn/(fn + tp) if (fn + tp) > 0 else None
        self.false_positive_rate.append(fpr)
        self.false_negative_rate.append(fnr)
        logger.info("## fpr: {}, fnr: {}".format(fpr, fnr))

        weight_accumulator = {}
        for name, data in self.target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
            for inlier in inliers:
                weight_accumulator[name].add_(self.local_models_weight_delta[inlier][name])
        return weight_accumulator

    def generate_diff_imgs2(self, epoch, n_iter = 20, s=0.5, lambda1=1, lambda2=0):
        #img_list = []; orig_img_list = []
        outlier_counter = defaultdict(list)
        losses = []
        for img_ind, seed in enumerate(self.original_input):
            input_img = seed
            input_label = self.original_label[img_ind]
            orig_img = input_img.detach().clone().unsqueeze(0).cuda()
            X = Variable(input_img, requires_grad=True).unsqueeze(0).cuda()

            for _ in range(n_iter):
                loss1, loss2, loss, outlier_models = self.compute_diff_weighted_obj(epoch, X, lambda1, lambda2)
                X.retain_grad()  ## check on this: why is this necessary?
                loss.backward()
                X.data += s*X.grad
                X.grad.zero_()

            X = X.detach()
            if outlier_models is not None:
                outlier_model_str = ",".join(outlier_models)
                diff_input_logger.info("{}, {}, {}, {}, {}, {}, {}".format(epoch, img_ind, input_label, round(loss1.item(),2),
                round(loss2.item(),2), round(loss.item(),2), outlier_model_str))
                final_loss = loss.item()
                losses.append(final_loss)
                for model in outlier_models:
                    outlier_counter[model].append(final_loss)

        return(losses, outlier_counter)

    def diff_input_aggregate2(self, losses, outlier_counter):
        # Based on the occurrence of outlier membership
        def mad_detect_outliers(dict, th=3):
            local_model_names = list(dict.keys())
            dict_vals = list(dict.values())
            med = np.median(dict_vals)
            abs_med_diff = abs(dict_vals-med)
            mad = np.median(abs_med_diff)
            mad = mad if abs(mad) > 0 else 1e-9
            mad_dist = abs_med_diff/mad
            outlier = [local_model_names[i] for i in range(len(mad_dist)) if mad_dist[i] > th]
            return outlier
        local_model_names = list(self.local_models_weight_delta.keys())
        outlier_counter = {int(k): np.mean(v) for k, v in outlier_counter.items()}
        inliers = set(local_model_names).difference(set(outlier_counter))
        for k in inliers:
            outlier_counter[k] = 0
        outliers = set(mad_detect_outliers(outlier_counter))
        inliers = set(local_model_names).difference(outliers)

        adversaries = set(self.params['adversary_list'])
        tp = len(outliers.intersection(adversaries)) # true positives
        fp = len(outliers) - tp  # false positives
        fn = len(inliers.intersection(adversaries)) # false negatives
        tn = len(inliers) - fn  # true negatives
        fpr = fp/(fp + tn) if (fp + tn) > 0 else None
        fnr = fn/(fn + tp) if (fn + tp) > 0 else None
        self.false_positive_rate.append(fpr)
        self.false_negative_rate.append(fnr)
        logger.info("## fpr: {}, fnr: {}".format(fpr, fnr))

        weight_accumulator = {}
        for name, data in self.target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)
            for inlier in inliers:
                weight_accumulator[name].add_(self.local_models_weight_delta[inlier][name])
        return weight_accumulator

    def foolsgold_aggregate(self):

        def elementwise_logit(x, confidence_param):
            return min(max(confidence_param * (np.log(x/(1-x)) + 0.5), 0.0), 1.0)

        local_model_names = list(self.local_models_weight_delta.keys())
        cs = np.zeros((len(local_model_names), len(local_model_names)))
        lrs = [0] * len(local_model_names)
        # Identify indicative features in the output layer (hard vs soft version?)
        target_output = torch.cat([self.target_model.state_dict()[x].view(-1) for x in self.params["mad_layer_names"]])
        n_ind_features = int(self.params["ind_feature_ratio"] * len(target_output))
        vals, inds = abs(target_output).topk(n_ind_features)

        # Retrive historical aggregate weights of current PCPs
        for i, cur_model in enumerate(local_model_names):
            cur_vec = self.historical_output_weights[cur_model]
            cur_vec = cur_vec[inds]
            #logger.info("!!!! cur model: {}, cur vec shape: {}".format(cur_model, cur_vec.shape))
            #other_model_names = list(set(local_model_names) - set(cur_model))
            for j, other_model in enumerate(local_model_names):
                if i==j:
                    cs[i,j] = -5
                else:
                    other_vec = self.historical_output_weights[other_model]
                    other_vec = other_vec[inds]
                    cs[i,j] = torch.nn.functional.cosine_similarity(cur_vec, other_vec, dim=0).cpu().tolist()

        max_cs = np.max(cs, axis=1)

        # Pardoning
        for i, cur_model in enumerate(local_model_names):
            vi = max_cs[i]
            for j, other_model in enumerate(local_model_names):
                vj = max_cs[j]
                if vj > vi:
                    cs[i,j] = vi/vj * cs[i,j]

            lrs[i] = 1 - np.max(cs[i,:])

        # Normalize learning rates
        lrs = lrs/max(lrs)
        # Element-wise logic function
        lrs = [elementwise_logit(x, self.params["logit_confidence"]) for x in lrs]
        # Weighted sum of local model delta
        weight_accumulator = {}
        for name, data in self.target_model.state_dict().items():
            #pdb.set_trace()
            weight_accumulator[name] = torch.zeros_like(data)
            if len(data.shape) == 0:
                continue
            for i, model_name in enumerate(local_model_names):
                weight_accumulator[name].add_(lrs[i] * self.local_models_weight_delta[model_name][name])

        # Save avg learning rate for adv and benign models
        adv_lrs = [lrs[x] for x in np.arange(len(lrs)) if local_model_names[x] in self.params["adversary_list"]]
        benign_lrs = [lrs[x] for x in np.arange(len(lrs)) if local_model_names[x] not in self.params["adversary_list"]]
        logger.info("FG: learning rates of adversaries: ".format(adv_lrs))
        logger.info("FG: learning rates of benign models: ".format(benign_lrs))
        #pdb.set_trace()
        if len(adv_lrs) > 0:
            self.adv_learning_rates.append(np.mean(adv_lrs))
        if len(benign_lrs) > 0:
            self.benign_learning_rates.append(np.mean(benign_lrs))

        return weight_accumulator

    # Save global model
    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model

        poison_epochs = self.params['poison_epochs']
        save_epochs = self.params['save_on_epochs'] + poison_epochs + [x-1 for x in poison_epochs] + [x+1 for x in poison_epochs]
        save_epochs = list(set(save_epochs))

        if self.params['save_model'] and epoch in save_epochs:
            # save_model
            logger.info("saving model")
            model_name = '{}/global_model_epoch_{}.pt.tar'.format(self.params['folder_path'], epoch)
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            #self.save_checkpoint(saved_dict, False, model_name)
            # By default, we save models during poison epochs, epochs right before/after poison epochs
            logger.info(f'Saving model on epoch {epoch}')
            self.save_checkpoint(saved_dict, False, filename=f'{model_name}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    # Save local model
    def save_local_model(self, model_id, model, epoch, val_loss, val_acc, adversary=False):
        if self.params['save_model']:
            poison_epochs = self.params['poison_epochs']
            # By default, we save models during poison epochs, epochs right before/after poison epochs
            save_epochs = self.params['save_on_epochs'] + poison_epochs + [x-1 for x in poison_epochs] + [x+1 for x in poison_epochs]
            save_epochs = list(set(save_epochs))
            if epoch not in save_epochs:
                return

            # Save local weight updates (delta between local and global models)
            weight_update_dict = {}
            for name, data in model.state_dict().items():
                weight_update_dict[name] = (data - self.target_model.state_dict()[name])

            if not os.path.exists(self.params['folder_path']):
                os.mkdir(self.params['folder_path'])
            if adversary:
                model_name = '{}/adversary_model_epoch_{}.pt.tar'.format(self.params['folder_path'], epoch)
                logger.info("Saving adversary model at epoch: {}".format(epoch))
            else:
                model_name = '{}/benign_model_{}_epoch_{}.pt.tar'.format(self.params['folder_path'], model_id, epoch)
                logger.info("Saving benign model at epoch: {}".format(epoch))

            saved_dict = {'state_dict': model.state_dict(), 'weight_update': weight_update_dict, 'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
            self.save_checkpoint(saved_dict, False, model_name)



    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                             evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )
