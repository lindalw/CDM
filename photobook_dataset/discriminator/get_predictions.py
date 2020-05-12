import json
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
from torch import nn
from models.model_nohistory import DiscriminatoryModelBlind
from models.model_history import HistoryModelBlind
from models.model_history_noimg import HistoryModelBlindNoTarget

import train_nohistory
import train_history
import train_history_noimg

from SegmentDataset import SegmentDataset
from HistoryDataset import HistoryDataset

from Vocab import Vocab

# from collections import defaultdict
import os

def load_model(file, model, device):
    len_vocab = 3424
    embedding_dim = 512
    hidden_dim = 512
    img_dim = 2048
    # Initialize model
    model = model(len_vocab, embedding_dim, hidden_dim, img_dim).to(device)
    # Get parameters
    checkpoint = torch.load(file, map_location=device)
    # Insert model parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    # Get models epoch, loss, acc and arguments
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    args_train = checkpoint['args']
    return model, epoch, loss, accuracy, args_train


def get_model(model_name, models_dict=False, device='cpu'):
    # Get model file name from the supplied or default pickle dict
    if not models_dict:
        models_dict = {'History':'model_history_blind_accs_2019-02-20-14-22-23.pkl',
                'No history': 'model_blind_accs_2019-02-17-21-18-7.pkl',
                    'No image': 'model_history_noimg_accs_2019-03-01-14-25-34.pkl'}
    model_file = models_dict[model_name]

    # Load model (with different models per modelname)
    if model_name=='No history':
        model, epoch, loss, accuracy, args_train = load_model(model_file, DiscriminatoryModelBlind, device)
    elif model_name == 'History':
        model, epoch, loss, accuracy, args_train = load_model(model_file, HistoryModelBlind, device)
    elif model_name == 'No image':
        model, epoch, loss, accuracy, args_train = load_model(model_file, HistoryModelBlindNoTarget, device)
    else:
        raise ValueError('Invalid model name')

    return model, epoch, loss, accuracy, args_train


def seg_rank_ids():
    """
    TODO: Get new seg2rank.json and seg_ids.json for new test.jsons from segment_ranks_ids.py
    For if we use new test json, might need new ordered ids for the history.
    Some stuff now commented out
    """
    with open('test_seg2ranks.json', 'r') as file:
        seg2ranks = json.load(file)

    with open('test_seg_ids.json','r') as file:
        id_list = json.load(file)  #ordered IDs for history

    # # Open the segments (containing encoded utterances)
    # # and chain file (containing the segments per chain and the target)
    # with open('data/test_segments.json', 'r') as file:
    #     test_sg = json.load(file)
    # with open('data/test_chains.json', 'r') as file:
    #     test_chain = json.load(file)
    # vocab = Vocab('data/vocab.csv', 3)

    # # given an img, provides the chains for which it was the target
    # # I.e. {'target_image_id': [segment_list1, segmentlist2...]}
    # target2chains = defaultdict(list)
    # for ch in test_chain:
    #     target_id = ch['target']
    #     segment_list = ch['segments']
    #     target2chains[target_id].append(segment_list)

    # # segments ids, in the order in which they were encountered in the chains in the whole dataset
    # id_list = []
    # # For each chain
    # for c in test_chain:
    #     segments = c['segments']
    #     # Add segment id to the id_list
    #     # Making sure there are no duplicates
    #     for s in segments:
    #         if s not in id_list:
    #             id_list.append(s)
    # with open('test_seg_ids.json', 'w') as file:
    #     json.dump(id_list, file)


    # seg2ranks = dict()
    # # For each chain
    # for c in test_chain:
    #     segments = c['segments']
    #     # For each segment
    #     for s in range(len(segments)):
    #         # Get id
    #         seg_id = segments[s]
    #         # And rank given in order of encountered in dataset
    #         rank = s
    #         if seg_id in seg2ranks:
    #             seg2ranks[seg_id].append(rank)
    #         else:
    #             seg2ranks[seg_id] = [rank]

    # # all the ranks a segment is positioned
    # # in different chains (hence, there could be multiples of the same position)
    # with open('test_seg2ranks.json', 'w') as file:
    #     json.dump(seg2ranks, file)

    return seg2ranks, id_list

def get_BCEloss_funct(weighting, pos_weight):
    if weighting:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')
    return criterion

def get_segment_dataset(batch_size=1, data_dir='data/', segment_file='segments.json', vectors_file='vectors.json', split='test', device='cpu'):
    """
    Get segment dataset that contains segments with:
    dataset['segment']: encoded segment (e.g. [3, 19, 5, 23])
    dataset['image_set']: 
    """
    testset = SegmentDataset(
        data_dir=data_dir,
        segment_file=segment_file,
        vectors_file=vectors_file,
        split=split
    )

    load_params_test = {'batch_size': batch_size,
                        'shuffle': False, 'collate_fn': SegmentDataset.get_collate_fn(device)}

    test_loader = torch.utils.data.DataLoader(testset, **load_params_test)
    return testset, test_loader


def get_history_dataset(batch_size=1, data_dir='data/', segment_file='test_segments.json', vectors_file='vectors.json', chain_file='test_chains.json', split='test', device='cpu'):
    """
    TODO: understand the history dataset
    """
    testset_hist = HistoryDataset(
        data_dir=data_dir,
        segment_file=segment_file,
        vectors_file=vectors_file,
        chain_file=chain_file,
        split=split
    )

    load_params_test_hist = {'batch_size': batch_size,
                        'shuffle': False, 'collate_fn': HistoryDataset.get_collate_fn(device)}


    test_hist_loader = torch.utils.data.DataLoader(testset_hist, **load_params_test_hist)
    return testset_hist, test_hist_loader


def get_predictions(model_name='History', models_dict=False):
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load model (with different models per modelname), either with supplied or default dict
    model, epoch, loss, accuracy, args_train = get_model(model_name=model_name, models_dict=models_dict, device=device)
    print('loaded model:', model_name)
    args = args_train

    # Load vocab
    vocab = Vocab(os.path.join(args.data_path, args.vocab_file), 3)
    print('vocab len', len(vocab))

    # TODO: Get new seg2rank.json and seg_ids.json for new test.jsons from segment_ranks_ids.py
    seg2ranks, id_list = seg_rank_ids()
    print('Loaded seg2ranks and idlist')

    # Get parameters? Why not just use args?
    # TODO: put in a dict?
    img_dim = 2048
    threshold = 0.5
    normalize = args.normalize
    mask = args.mask
    weight = args.weight
    weights = torch.Tensor([weight]).to(device)
    batch_size = 1
    print(f"params. normalize={args.normalize}, mask={args.mask}, weight={args.weight}, weighting={args.weighting}, batchsize={batch_size}, breaking={args.breaking}")

    # Get (weighted) loss function
    criterion = get_BCEloss_funct(weighting=args.weighting, pos_weight=weights)

    # Get segment dataset
    print(f"Dataparams. data_dir={args.data_path}, segmentfile={args.segment_file}, vectorfile={args.vectors_file}, chains_file={args.chains_file}")
    # testset, test_loader = get_segment_dataset(data_dir='data/', segment_file='segments.json', vectors_file='vectors.json', split='test')
    testset, test_loader = get_segment_dataset(batch_size=batch_size, data_dir=args.data_path, segment_file=args.segment_file,
                                                 vectors_file=args.vectors_file, split='test', device=device)

    # Get history dataset
    # testset_hist, test_hist_loader = get_history_dataset(batch_size=1, data_dir='data/', segment_file='test_segments.json', vectors_file='vectors.json', chain_file='test_chains.json', split='test', device='cpu')
    testset_hist, test_hist_loader = get_history_dataset(batch_size=batch_size, data_dir=args.data_path, segment_file='test_' + args.segment_file,
                                                         vectors_file=args.vectors_file, chain_file='test_' + args.chains_file, 
                                                         split='test', device=device)

    # TODO: Change eval to just the predictions
    dataset_pred = False
    with torch.no_grad():
        model.eval()
        print('\nGold Eval')

        if model_name == 'No history':
            print('predict no history')
            # The predict function returns the dataset with added predicions, loss and rank
            # rank_p_1, rank_r_1, rank_p_0, rank_r_0, segment_rank_res = train_nohistory.gold_evaluate(test_loader, testset, args.breaking, normalize, mask, img_dim, model, seg2ranks, device, criterion, threshold, weight)
            dataset_pred = train_nohistory.predict(test_loader, testset, args.breaking, normalize, mask, img_dim, model, seg2ranks, device, criterion, threshold, weight)
            print(dataset_pred[0])
        elif model_name == 'History':
            # TODO: make prediction function (see no history predict)
            rank_p_1, rank_r_1, rank_p_0, rank_r_0, segment_rank_res = train_history.gold_evaluate(test_hist_loader, testset_hist, args.breaking, normalize, mask, img_dim, model, seg2ranks, id_list, device, criterion, threshold, weight)
        elif model_name == 'No image':
            # TODO: make prediction function (see no history predict)
            rank_p_1, rank_r_1, rank_p_0, rank_r_0, segment_rank_res = train_history_noimg.gold_evaluate(test_hist_loader, testset_hist, args.breaking, normalize, mask, img_dim, model, seg2ranks, id_list, device, criterion, threshold, weight)

    return dataset_pred


if __name__ == '__main__':

    models_dict = {'History':'model_history_blind_accs_2019-02-20-14-22-23.pkl',
                'No history': 'model_blind_accs_2019-02-17-21-18-7.pkl',
                    'No image': 'model_history_noimg_accs_2019-03-01-14-25-34.pkl'}

    dataset_pred = get_predictions(model_name='No history', models_dict=models_dict)
    breakpoint()
    print('done')