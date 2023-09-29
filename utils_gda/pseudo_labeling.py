import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
from utils_gda.validate import accuracy
from adversarial.attack import attackstep
from utils_gda import tsne
import copy

def pseudo_labeling(classifier_model, cur_x: torch.tensor, cur_y: torch.tensor, device, vis_path, t, epsilon, confi_q = 0.10, visualize = False):
    """
    input:
        classifier_model:  the trained classfier model used to classify which class it belongs to
        cur_x:  the features of the current domain, shape: [interval_size, channel, image_h, image_w]
        cur_y:  the labels of the current domain, shape: [interval_size, ]
        vis_path:  the save path of the T-SNE figures
        confi_q:  the confidence level for conformal prediction
        visualize=True:  whether visualize the features by T-SNE
        noisy_train=True:  whether conduct the experiments on randomly noisy data
    output:
        pool_x_pseudo: the filtered datas. It is a torch.Tensor data. shape: [*, channel, image_h, image_w]
        pool_y_pseudo: the pseudo label set predicted by predictor. It is a torch.Tensor data. shape: [*, num_classes]
        labeling_acc: use the cur_y to validate the accuracy of the conformal predictor
        labeling_size: the average size of label set
    """
    classifier_model.eval()
    pool_x = []
    pool_prediction = []
    pool_features = []
    pool_y_pseudo = []
    pool_label = []
    pool_confidence = []
    pool_attackstep = []

    # labeling_acc = 0
    dataset = TensorDataset(cur_x, cur_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for _, data in enumerate(dataloader):
        x, label = data
        x = x.to(device)
        label = label.to(device)
        predictions, features = classifier_model(x, True)  # return the output of the intermediate layers
        max_logits, pred_logits = torch.max(predictions, 1)
        min_logits, _ = torch.min(predictions, 1)
        confidence = max_logits - min_logits
        attack_step = attackstep(classifier_model, x, epsilon) if visualize else torch.zeros_like(label)  # return the minimal steps of successful PGD attack for each sample.
        if len(pool_y_pseudo) == 0:
            pool_y_pseudo = pred_logits
            pool_prediction = predictions
            pool_x = x
            pool_confidence = confidence
            pool_label = label
            pool_features = features
            pool_attackstep = attack_step
        else:
            pool_y_pseudo = torch.cat((pool_y_pseudo, pred_logits), 0)
            pool_x = torch.cat((pool_x, x), 0)
            pool_confidence = torch.cat((pool_confidence, confidence), 0)
            pool_label = torch.cat((pool_label, label), 0)
            pool_prediction = torch.cat((pool_prediction, predictions), 0)
            pool_features = torch.cat((pool_features, features), 0)
            pool_attackstep = torch.cat((pool_attackstep, attack_step), 0)
        # acc = accuracy(predictions, label)
        # labeling_acc += acc
    pool_features = pool_features.detach().cpu()

    quantile_point = torch.quantile(pool_confidence, confi_q)
    index_remained = pool_confidence > quantile_point  # index_filter indicates whether the data is filtered out with high confidence.
    labeling_acc, index_correct = accuracy(pool_prediction, pool_label, True) # index_correct indicates whether the data has correct pseudo-label.
    labeling_acc_filtered = accuracy(pool_prediction[index_remained], pool_label[index_remained])

    if visualize:
        # plot t-SNE
        tSNE_filename = 'TSNE' + str(t) + '.pdf'
        save_path = os.path.join(vis_path, tSNE_filename)
        correct_features = pool_features[index_correct]
        wrong_remained_features = pool_features[~index_correct & index_remained]
        wrong_filtered_features = pool_features[~index_correct & ~index_remained]
        all_features = torch.cat((correct_features, wrong_remained_features, wrong_filtered_features), 0)
        all_domains = torch.cat((torch.zeros(correct_features.shape[0]), torch.ones(wrong_remained_features.shape[0]), 2*torch.ones(wrong_filtered_features.shape[0])), 0)
        tsne.visualize(all_features, all_domains, save_path)
        print("Saving t-SNE to", save_path)

        # calculate the average attack step numbers
        avgstep_correct = sum(pool_attackstep[index_correct]) / sum(index_correct)
        avgstep_wrong = sum(pool_attackstep[~index_correct]) / sum(~index_correct)
        print('The average attack steps of correct data is:', avgstep_correct)
        print('The average attack steps of wrong data is:', avgstep_wrong)

    return pool_x[index_remained], pool_label[index_remained], pool_y_pseudo[index_remained], labeling_acc.item(), labeling_acc_filtered.item(), sum(index_remained)
