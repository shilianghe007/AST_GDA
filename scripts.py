
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import copy

from data.datasets import load_data, load_transform, data_property
from models.create_model import create_model
from train_model import train_model
from utils_gda.args_utils import create_args
from adversarial.validate import validate
from utils_gda.logger import CompleteLogger
from utils_gda.pseudo_labeling import pseudo_labeling
from utils_gda.noisy import noisify_label
import torch
import numpy as np
from torch.optim import SGD
import random
from utils_gda.save_utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def main(args):
    if args.oracle:
        log_path = "{0}/{1}/epoch_{2}_noisy_{3}_lr_{4}_confi_{5}_epsilon_{6}_oracle_{7}/starttime{8}".format(args.log_file, args.data, args.epoches, args.noisy_sigma, args.lr, args.confi_q, args.epsilon, args.oracle, args.ad_time)
    else:
        log_path = "{0}/{1}/epoch_{2}_noisy_{3}_lr_{4}_confi_{5}_epsilon_{6}/starttime{7}".format(args.log_file, args.data, args.epoches, args.noisy_sigma, args.lr, args.confi_q, args.epsilon, args.ad_time)
    logger = CompleteLogger(log_path)
    writer = SummaryWriter(os.path.join(log_path, "visualize"))
    print(args)

    ## load data
    print("Loading data...")
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, _, _,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = load_data(args.data)
    transform = load_transform(args.data) # the argmentation of dataset for training

    ## record the best acc
    best_acc = 0
    best_acc_ad = 0
    accs = []
    accs_ad = []
    ## classifier model
    print("Loading classifier...")
    classifier_model = create_model(args.data, args.num_classes, args.arch)

    ## test classifier on source domain (the random initial model)
    print('validate classifier on target domain:')
    val_acc, val_acc_ad, val_loss_ad = validate(classifier_model, (trg_val_x, trg_val_y), args, args.epsilon, print_result=True)
    writer.add_scalar('target_loss_ad', val_loss_ad, -1)
    writer.add_scalar('target_acc', val_acc, -1)
    writer.add_scalar('target_acc_ad', val_acc_ad, -1)
    accs.append(val_acc)
    accs_ad.append(val_acc_ad)
    ## define optimizer and lr scheduler
    optimizer = SGD(classifier_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    ## train classifier on source domain
    print("Start training source model on source domain...")
    epsilon_train = args.epsilon if args.ad_time <= 0 else 0
    src_tr_x = src_tr_x + torch.randn_like(src_tr_x)*args.noisy_sigma if args.ad_time == 0 else src_tr_x
    train_acc, train_acc_ad = train_model((src_tr_x, src_tr_y, src_val_x, src_val_y), transform, classifier_model, epsilon_train, optimizer, args.epoches, args.batch_size, writer, 0, args)
    ## test classifier on target domain
    print('validate classifier on target domain:')
    val_acc, val_acc_ad, val_loss_ad = validate(classifier_model, (trg_val_x, trg_val_y), args, args.epsilon, print_result=True)
    writer.add_scalar('target_loss_ad', val_loss_ad, 0)
    writer.add_scalar('target_acc', val_acc, 0)
    writer.add_scalar('target_acc_ad', val_acc_ad, 0)
    accs.append(val_acc)
    accs_ad.append(val_acc_ad)
    print("Finish training source model on source domain!\n")


    ## gradual self-training
    interval = args.interval
    intermediate_domain = int(inter_x.shape[0] / interval)
    for t in range(1, intermediate_domain + 1):
        print("\nThe {}th intermediate domain!".format(t))

        # clone a copy version of the classifier model for the control experiments on randomly noisy label
        if args.noisy_train:
            model = copy.deepcopy(classifier_model)

        ## split the dataset
        cur_x = inter_x[interval*(t-1):interval*t]
        cur_y = inter_y[interval*(t-1):interval*t]

        ## labeling the intermediate domain
        vis_path = os.path.join(log_path, "visualize")
        cur_x_filter, cur_y_filter, cur_y_pseudo, labeling_acc, labeling_acc_filtered, labeling_size = pseudo_labeling(classifier_model, cur_x, cur_y, device, vis_path, t, args.epsilon, args.confi_q, args.visualize) # cur_y_pseudo is the pseudo label set, labeling_acc is the labeling accuracy and alpha is the confidence level
        print('The acc of pseudo label: ', labeling_acc, 'The acc of pseudo label on filtered data: ', labeling_acc_filtered, ' The average size of label set: ', labeling_size)
        
        ## finetune on the pseudo label
        print("Finetune the classifier on the generated data...")
        epsilon_train = args.epsilon if t >= args.ad_time else 0
        cur_x_filter = cur_x_filter + torch.randn_like(cur_x_filter)*args.noisy_sigma if t >= args.ad_time else cur_x_filter
        optimizer = SGD(classifier_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        if args.oracle == True:
            train_acc, train_acc_ad = train_model((cur_x, cur_y, cur_x, cur_y), transform, classifier_model, epsilon_train, optimizer, args.epoches, args.batch_size, writer, t, args)  # using the ground truth label to finetune the model
        elif args.oracle == False:
            train_acc, train_acc_ad = train_model((cur_x_filter, cur_y_pseudo, cur_x, cur_y), transform, classifier_model, epsilon_train, optimizer, args.epoches, args.batch_size, writer, t, args)  # cur_x and cur_y are the true data and label of the intermediate data

        ## test classifier on target domain
        print('validate classifier on target domain:')
        val_acc, val_acc_ad, val_loss_ad = validate(classifier_model, (trg_val_x, trg_val_y), args, args.epsilon, print_result=True)
        writer.add_scalar('target_loss', val_loss_ad, t)
        writer.add_scalar('target_acc', val_acc, t)
        writer.add_scalar('target_acc_ad', val_acc_ad, t)
        accs.append(val_acc)
        accs_ad.append(val_acc_ad)
        print('accs:', accs)
        print('accs_ad:', accs_ad)
        if best_acc <= val_acc:
            best_acc = val_acc
            model_best = copy.deepcopy(classifier_model)
        if best_acc_ad <= val_acc_ad:
            best_acc_ad = val_acc_ad
            model_best_ad = copy.deepcopy(classifier_model)
        
        # finetune on the randomly noisy label (a control experiment)
        if args.noisy_train:
            print("Control experiments on randomly noisy label...")
            model_clean = model # be used for standard training
            model_adv = copy.deepcopy(model_clean)  # be used for adversarial training
            # randomly generate noisy label with a specific proposion which is the same as the proposion of noisy label in the fileted training data of the next domain. We aim to show that AT only works on specific noisy label data.
            train_noisy_y = noisify_label(cur_y_filter.cpu(), 'symmetric', 1-labeling_acc_filtered, args.num_classes)
            # standard training
            optimizer = SGD(model_clean.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            train_acc, train_acc_ad = train_model((cur_x_filter, train_noisy_y, cur_x, cur_y), transform, model_clean, 0, optimizer, args.epoches, args.batch_size, writer, -1, args, present_progress=False)
            print('The acc and acc_ad of the standardly trained model are: %.2f%% and %.2f%%' % (train_acc*100, train_acc_ad*100))
            # adversarial training
            optimizer = SGD(model_adv.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            train_acc, train_acc_ad = train_model((cur_x_filter, train_noisy_y, cur_x, cur_y), transform, model_adv, args.epsilon, optimizer, args.epoches, args.batch_size, writer, -1, args, present_progress=False)
            print('The acc and acc_ad of the adversarially trained model are: %.2f%% and %.2f%%' % (train_acc*100, train_acc_ad*100))


    ## test the best model on target domain
    best_acc, _, _ = validate(model_best, (trg_test_x, trg_test_y), args, args.epsilon)
    _, best_acc_ad, _ = validate(model_best_ad, (trg_test_x, trg_test_y), args, args.epsilon)
    print("The test acc/acc_ad of the best model on target domain: ", best_acc, '/', best_acc_ad)
    ## save the best model
    save_checkpoint(
        state={
            'args': args,
            'accs': accs,
            'best_acc': best_acc,
            'model_best': model_best,
            'model_best_ad': model_best_ad,
            'model_final': classifier_model,
        },
        checkpoint=os.path.join(log_path, "checkpoints"),
        filename='checkpoint.pth.tar'
    )

if __name__ == "__main__":
    os.chdir(sys.path[0])
    ## load arguments
    special_args = dict(
        log_file='logs_test_rebuttal',  # the file location of the logs, '0.01' denotes the step size of attacker.
        data='mnist',  # mnist or portraits
        batch_size=32, # classifier training batchsize
        interval=2000, # the number of data in an intermediate domain
        confi_q=0.05,  # confidence score of pseudo labeling
        epsilon=0.1,   # the adversarial perturbation
        per_class_eval=0,   # 0: No  1: Yes, evaluate the accuracy for each class
        ad_time=0,     
        epoches = 40,
        noisy_sigma = 0, # the variance of the added noisy distribution
        arch='3conv', # 3conv is the same network as the original implement. resnet18, resnet34, and resnet50 are supported.
        lr=0.001,
        dy_lr=0,  # False for mnist; True for cifar
        weight_decay=1e-3,
        oracle = False,  # whether use the groundtruth label of the intermediate domains for training
        visualize=False, # whether visualize the features by T-SNE
        noisy_train=False,  # whether conduct the experiments on randomly noisy data
    )
    parser = create_args(special_args)
    args = parser.parse_args()
    args.num_classes, args.image_size = data_property(args.data) # load the dataset properties
    
    ## main
    setup_seed(0)
    main(args)