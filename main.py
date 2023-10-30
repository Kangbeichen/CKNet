import argparse

import numpy as np
import torch

from models.models import BaseCMNModel
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from torch.utils.tensorboard.writer import SummaryWriter
import os



def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/Users/leslie/data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    # parser.add_argument('--image_dir', type=str, default='/home/jiaxing/data/iu_xray/images/',
    #                     help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/Users/leslie/data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')
    # parser.add_argument('--ann_path', type=str, default='/home/jiaxing/data/iu_xray/annotation.json',
    #                     help='the path to the directory containing the data.')
    # parser.add_argument('--label_path', type=str, default='/home/jiaxing/data/iu_xray/labels_14.pickle',
    #                     help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')    #16

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=16, help='the number of k.')#32
    # parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')  #100
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=10, help='the logging interval (in batches).')  #1000
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    #1:lr_ve: 5e-05  lr_ed: 0.0007----不稳定，最好17
    #2:1e-4  5e-4
    parser.add_argument('--lr_ve', type=float, default=1e-4, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=5e-4, help='the learning rate for the remaining parameters.')  #7e-4
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    #KG
    parser.add_argument('--num-classes', type=int, default=30)
    #Current best Epoch: 80, best auc: 0.7571157698365678,,----=用80的反而不行
    parser.add_argument('--pretrained', type=str, default= './gcnclassifier_30keywords_e80.pth')


    args = parser.parse_args()
    return args


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #在7号GPU上
    # iu_xray:batch_size 8----单卡5427M 259step
    # iu_xray:batch_size 16----单卡8341M 130step

    # parse arguments
    args = parse_agrs()
    print('------------------------Model and Training Details--------------------------')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    # 加上
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    print("---------------------create data loader------------------")
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)


    #build KG   先验知识
    with open('./data/openi_top30/auxillary_openi_matrix_30nodes.txt', 'r') as matrix_file:
        adjacency_matrix = [[int(num) for num in line.split(', ')] for line in matrix_file]
    fw_adj = torch.tensor(adjacency_matrix, dtype=torch.float, device=device)  # (31,31)
    bw_adj = fw_adj.t()
    identity_matrix = torch.eye(args.num_classes + 1, device=device)  # (31,31)
    fw_adj = fw_adj.add(identity_matrix)  # (31,31)   加上自连接，A波浪
    bw_adj = bw_adj.add(identity_matrix)

    # build model architecture
    print("---------------------build model architecture------------------")
    model = BaseCMNModel(args, fw_adj, bw_adj, tokenizer)

    # fixed the parameters in the graph embedding module
    for param in model.gcn.parameters():
        param.requires_grad = False
    model.gcn.eval()

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    if args.pretrained:
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['model_state_dict']
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in pretrained_state_dict.items() if k in state_dict and 'fc' not in k})
        model.load_state_dict(state_dict)
        print('pre-trained parameters are loaded!')

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    print("---------------------build trainer and start to train------------------")
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
