import copy
import pandas as pd
import json
from tqdm import tqdm
import numpy as np

from get_predictions import get_predictions
from ChainDataset import ChainDataset

def get_seg_ids(segment_ids_file='segment_ids_test.json'):
    # Load in the segment ids that tell us which segment belongs to which index in the history dataset
    with open(segment_ids_file) as json_file:
        segment_ids = json.load(json_file)

    return segment_ids

def create_inv_list(segment_ids):
    # Get inverted dict {seg_id:dataset_ind}
    inv_dict = {}
    for data_i in range(len(segment_ids)):
        inv_dict[segment_ids[data_i]] = data_i
    # Sort it based on segmentids
    keys = list(inv_dict)
    keys.sort()
    inv_list = [inv_dict[key] for key in keys]

    return inv_list

def reorder_datast(dataset_pred_hist, inv_list):
    # Create new history dataset with the segments in the same order as the no-history dataset
    dataset_pred_hist_cp = copy.deepcopy(dataset_pred_hist)
    # For each segment_id replace the hist_cp with the corresponding data from dataset_pred_hist
    for i in range(len(inv_list)):
        # Replace each key (cause you can't replace entire dicts here apparently...)
        for key in list(dataset_pred_hist[i]):
            dataset_pred_hist_cp[i][key] = dataset_pred_hist[inv_list[i]][key]

    return dataset_pred_hist_cp


def get_pred_dataframe(dataset_pred_no_hist, dataset_pred_hist_cp):
    """Get dataframe with the predictions per segment id of the history or no history dataset"""
    datasets = {"No history":dataset_pred_no_hist, "History": dataset_pred_hist_cp}
    dataframe = {i:{} for i in range(len(dataset_pred_no_hist))}
    print(dataset_pred_hist_cp[1]['segment'], dataset_pred_no_hist[1]['segment'])

    for data_key in datasets:
        print(data_key)
        dataset = datasets[data_key]
        # loop through all 6801 segments aka predictions
        for i in tqdm(range(len(dataset))):
            # If there is only one target and not more than 1 prediction
            if (len(dataset[i]['targets']) == 1 and
                dataset[i]['preds'].sum() <= 1):
                # If it is wrongly predicted or no image was predicted
                if dataset[i]['preds'].argmax() != dataset[i]['targets'][0] or dataset[i]['preds'].max()==0:
                    dataframe[i][data_key]=0
                # If target and prediction are the same, set prediction at 1
                else:
                    dataframe[i][data_key]=1
    return dataframe


def get_conditions_inds(dataframe):
    """Sort segment indices of the dataframe with history and no history model in the conditions"""
    # History, no history
    conditions_inds = {'hT_nhT':[], 'hT_nhF':[], 'hF_nhT':[], 'hF_nhF':[], 'only_h':[], 'only_nh':[], 'nothing':[], 'all':[], 'h_nh_all':[]}
    for ind in dataframe:
        # For all indices with the same
        if 'History' in dataframe[ind] and 'No history' in dataframe[ind]:
            if dataframe[ind]['History'] == 1 and dataframe[ind]['No history'] == 1:
                conditions_inds['hT_nhT'].append(ind)

            elif dataframe[ind]['History'] == 1 and dataframe[ind]['No history'] == 0:
                conditions_inds['hT_nhF'].append(ind)

            elif dataframe[ind]['History'] == 0 and dataframe[ind]['No history'] == 1:
                conditions_inds['hF_nhT'].append(ind)

            elif dataframe[ind]['History'] == 0 and dataframe[ind]['No history'] == 0:
                conditions_inds['hF_nhF'].append(ind)
            # For all segments that were predicted in history and no history
            conditions_inds['h_nh_all'].append(ind)

        elif 'History' in dataframe[ind]:
            conditions_inds['only_h'].append(ind)

        elif 'No history' in dataframe[ind]:
            conditions_inds['only_nh'].append(ind)
        else:
            conditions_inds['nothing'].append(ind)
        # For all segments together
        conditions_inds['all'].append(ind)

    return conditions_inds


def add_chains_rounds(dataset_pred_no_hist, dataset_pred_hist_cp, chain_test_set):
    """
    Create a dataset dictionary with per segment the chain ids to which it belongs, including
    the round in that chain that is occurs in and all the segment ids of that chain
    """
    
    # Create empty lists for the chain ids
    for data_ind in range(len(dataset_pred_hist_cp)):
        dataset_pred_hist_cp[data_ind]['chains'] = []
        dataset_pred_hist_cp[data_ind]['chain_hist'] = []
        dataset_pred_hist_cp[data_ind]['rounds'] = []

    for data_ind in range(len(dataset_pred_no_hist)):
        dataset_pred_no_hist[data_ind]['chains'] = []
        dataset_pred_no_hist[data_ind]['chain_hist'] = []
        dataset_pred_no_hist[data_ind]['rounds'] = []

    # For each chain, add the chain index to the segment id in the dataset to which they belong
    for chain_ind in range(len(chain_test_set.chains)):
        seg_ids = chain_test_set.chains[chain_ind]['segments']
        for i in range(len(seg_ids)):
            # Add chain index
            dataset_pred_no_hist[seg_ids[i]]['chains'].append(chain_ind)
            dataset_pred_hist_cp[seg_ids[i]]['chains'].append(chain_ind)

            # Add the segment ids in the chain
            dataset_pred_no_hist[seg_ids[i]]['chain_hist'].append(chain_test_set.chains[chain_ind]['segments'])
            dataset_pred_hist_cp[seg_ids[i]]['chain_hist'].append(chain_test_set.chains[chain_ind]['segments'])

            # Add the round of the segment in the chain
            dataset_pred_no_hist[seg_ids[i]]['rounds'].append(i)
            dataset_pred_hist_cp[seg_ids[i]]['rounds'].append(i)

    return dataset_pred_no_hist, dataset_pred_hist_cp

def get_condition_seg_hist(conditions_inds, dataset_pred_hist_cp):
    condition_seg_hist = {}
    # For each condition
    for condition in conditions_inds:
        condition_seg_hist[condition] = {}
        # Iterate over the indices in this condition
        for seg_id in conditions_inds[condition]:
            condition_seg_hist[condition][seg_id] = {}
            # For each chain that this segment belongs to
            # if the segment is not the first round in that chain
            for chain_i in range(len(dataset_pred_hist_cp[seg_id]['chains'])):
                if dataset_pred_hist_cp[seg_id]['rounds'][chain_i] > 0:
                    chain_ind = dataset_pred_hist_cp[seg_id]['chains'][chain_i]
                    condition_seg_hist[condition][seg_id][chain_ind] = {}
                    # Add first segment, and segment
                    first_id = dataset_pred_hist_cp[seg_id]['chain_hist'][chain_i][0]
                    condition_seg_hist[condition][seg_id][chain_ind]['first_id'] = first_id
                    condition_seg_hist[condition][seg_id][chain_ind]['first_seg'] = dataset_pred_hist_cp[first_id]['segment']
                    condition_seg_hist[condition][seg_id][chain_ind]['current_id'] = seg_id
                    condition_seg_hist[condition][seg_id][chain_ind]['current_seg'] = dataset_pred_hist_cp[seg_id]['segment']
                    condition_seg_hist[condition][seg_id][chain_ind]['round'] = dataset_pred_hist_cp[seg_id]['rounds'][chain_i]

    return condition_seg_hist


def get_pred_datasets_orig(split='test'):
    """
    Get the dataset of predictions for the experiment 'split'.
    Using the experiment datafiles; {split}_segments.json, {split}_segments.json
    """
    models_dict = {'History':'model_history_blind_accs_2019-02-20-14-22-23.pkl',
                'No history': 'model_blind_accs_2019-02-17-21-18-7.pkl',
                    'No image': 'model_history_noimg_accs_2019-03-01-14-25-34.pkl'}

    # Get predictions for this experiment (split) files
    dataset_pred_no_hist = get_predictions(model_name='No history', models_dict=models_dict, split=split)

    # History dataset takes about 20 minutes to run
    dataset_pred_hist = get_predictions(model_name='History', models_dict=models_dict, split=split)

    # Load in the segment ids that tell us which segment belongs to which index in the history dataset

    # Segment_ids_file segment_ids_test.json is created when predicting dataset_pred_hist
    segment_ids = get_seg_ids(segment_ids_file='segment_ids_test.json')

    # Get inverted dict {seg_id:dataset_ind}
    inv_list = create_inv_list(segment_ids)

    # Create new history dataset with the segments in the same order as the no-history dataset
    dataset_pred_hist_cp = reorder_datast(dataset_pred_hist, inv_list)

    # Add the chain indices and the round in those chains per segement
    chain_test_set = ChainDataset(
        data_dir='data/',
        segment_file='segments.json',
        chain_file='test_chains.json',
        vectors_file='vectors.json',
        split=split
    )
    # Get dataset with for each
    dataset_pred_no_hist, dataset_pred_hist_cp = add_chains_rounds(dataset_pred_no_hist, dataset_pred_hist_cp, chain_test_set)

    dataframe = get_pred_dataframe(dataset_pred_no_hist, dataset_pred_hist_cp)
    conditions_inds = get_conditions_inds(dataframe)

    condition_seg_hist = get_condition_seg_hist(conditions_inds, dataset_pred_hist_cp)

    return dataset_pred_no_hist, dataset_pred_hist_cp, conditions_inds, condition_seg_hist, dataframe


def get_pred_datasets(split='test'):
    """
    Get the dataset of predictions for the experiment 'split'.
    Using the experiment datafiles; {split}_segments.json, {split}_segments.json
    """
    models_dict = {'History':'model_history_blind_accs_2019-02-20-14-22-23.pkl',
                'No history': 'model_blind_accs_2019-02-17-21-18-7.pkl',
                    'No image': 'model_history_noimg_accs_2019-03-01-14-25-34.pkl'}

    # Get predictions for this experiment (split) files
    dataset_pred_no_hist = get_predictions(model_name='No history', models_dict=models_dict, split=split)

    # History dataset takes about 20 minutes to run
    dataset_pred_hist = get_predictions(model_name='History', models_dict=models_dict, split=split)

    # Load in the segment ids that tell us which segment belongs to which index in the history dataset

    # Segment_ids_file segment_ids_test.json is created when predicting dataset_pred_hist
    segment_ids = get_seg_ids(segment_ids_file='segment_ids_test.json')

    # Get inverted dict {seg_id:dataset_ind}
    inv_list = create_inv_list(segment_ids)

    # Create new history dataset with the segments in the same order as the no-history dataset
    dataset_pred_hist_cp = reorder_datast(dataset_pred_hist, inv_list)

    # Add the chain indices and the round in those chains per segement
    chain_test_set = ChainDataset(
        data_dir='data/',
        segment_file='segments.json',
        chain_file='test_chains.json',
        vectors_file='vectors.json',
        split=split
    )
    # Get dataset with for each
    dataset_pred_no_hist, dataset_pred_hist_cp = add_chains_rounds(dataset_pred_no_hist, dataset_pred_hist_cp, chain_test_set)

    # dataframe = get_pred_dataframe(dataset_pred_no_hist, dataset_pred_hist_cp)
    # conditions_inds = get_conditions_inds(dataframe)

    # condition_seg_hist = get_condition_seg_hist(conditions_inds, dataset_pred_hist_cp)

    return dataset_pred_no_hist, dataset_pred_hist_cp


def get_accuracies(conditions_inds, dataframe):
    """Returns the accuracies per condition"""
    results_hist = {condition:[] for condition in conditions_inds}
    results_nohist = {condition:[] for condition in conditions_inds}

    # Now for all segements, also those without history (i.e. in round 1)
    # Add 1 to the condition list if it was correctly predicted, 0 if incorrect
    for condition in conditions_inds:
        for ind in conditions_inds[condition]:
            if "History" in dataframe[ind]:
                results_hist[condition].append(dataframe[ind]['History'])
            if "No history" in dataframe[ind]:
                results_nohist[condition].append(dataframe[ind]['No history'])

    # Get the accuracies per condition
    accs_hist = {condition:[] for condition in conditions_inds}
    accs_nohist = {condition:[] for condition in conditions_inds}
    for condition in results_hist:
        res = np.array(results_hist[condition])
        accs_hist[condition] = res.sum()/len(res)
        res = np.array(results_nohist[condition])
        accs_nohist[condition] = res.sum()/len(res)

    return results_hist, results_nohist, accs_hist, accs_nohist


def get_accuracies_seg(conditions_inds, dataframe, changed_seg_ids):
    """Returns the accuracies per condition
    Only uses segments with their id in changed_seg_ids"""
    results_hist = {condition:[] for condition in conditions_inds}
    results_nohist = {condition:[] for condition in conditions_inds}

    # Now for all segements, also those without history (i.e. in round 1)
    # Add 1 to the condition list if it was correctly predicted, 0 if incorrect
    for condition in conditions_inds:
        for ind in conditions_inds[condition]:
            # Only use the segments that have a changed history
            if ind not in changed_seg_ids:
                continue
            if "History" in dataframe[ind]:
                results_hist[condition].append(dataframe[ind]['History'])
            if "No history" in dataframe[ind]:
                results_nohist[condition].append(dataframe[ind]['No history'])
                
    # Get the accuracies per condition
    accs_hist = {condition:[] for condition in conditions_inds}
    accs_nohist = {condition:[] for condition in conditions_inds}
    for condition in results_hist:
        res = np.array(results_hist[condition])
        accs_hist[condition] = res.sum()/len(res)
        res = np.array(results_nohist[condition])
        accs_nohist[condition] = res.sum()/len(res)
    
    return results_hist, results_nohist, accs_hist, accs_nohist


def get_img_dict(chain_test_set):
    """
    Return a dictionary of
    {img_id:{'round_id':'segments':[segids],
                        'lengths':[seglengths],
                        'game_ids':[gameids]}}
    where img_id is the image id of the target of the segments
    round_id is the round/rank, i.e. the i'th time the image is being talked about
    segid is the segment id
    seglength is the length of the segment with the corresponding index
    gameid is the gameid to which the segment with the corresponding index belongs
    """
    img_dict = {}
    chains = chain_test_set.chains
    # Get segments, lengths and gameids
    for chain_id in range(len(chains)):
        image_ind = chains[chain_id]['target']
        game_id = chains[chain_id]['game_id']
        # Add target image index to the img_dict
        if image_ind not in img_dict:
            img_dict[image_ind] = {}

        # Iterate over the rounds/segments
        for round_ind in range(len(chains[chain_id]['segments'])):

            # Add round to the dict for this target image
            if round_ind not in img_dict[image_ind]:
                img_dict[image_ind][round_ind] = {'segments':[],
                                                  'lengths':[],
                                                 'game_ids':[]}

            # Add segment with the length and gameid info to this round dict
            img_dict[image_ind][round_ind]['segments'].append(chains[chain_id]['segments'][round_ind])
            img_dict[image_ind][round_ind]['lengths'].append(chains[chain_id]['lengths'][round_ind])
            img_dict[image_ind][round_ind]['game_ids'].append(game_id)
    return img_dict


def get_img_dict(chains):
    """
    chains is chain_test_set.chains
    Return a dictionary of
    {img_id:{'round_id':'segments':[segids],
                        'lengths':[seglengths],
                        'game_ids':[gameids]}}
    where img_id is the image id of the target of the segments
    round_id is the round/rank, i.e. the i'th time the image is being talked about
    segid is the segment id
    seglength is the length of the segment with the corresponding index
    gameid is the gameid to which the segment with the corresponding index belongs
    """
    img_dict = {}
    # Get segments, lengths and gameids
    for chain_id in range(len(chains)):
        image_ind = chains[chain_id]['target']
        game_id = chains[chain_id]['game_id']
        # Add target image index to the img_dict
        if image_ind not in img_dict:
            img_dict[image_ind] = {}

        # Iterate over the rounds/segments
        for round_ind in range(len(chains[chain_id]['segments'])):

            # Add round to the dict for this target image
            if round_ind not in img_dict[image_ind]:
                img_dict[image_ind][round_ind] = {'segments':[],
                                                  'lengths':[],
                                                 'game_ids':[],
                                                 'chain_ids':[],
                                                 'chain_ids_shuf':[]}

            # Add segment with the length and gameid info to this round dict
            img_dict[image_ind][round_ind]['segments'].append(chains[chain_id]['segments'][round_ind])
            img_dict[image_ind][round_ind]['lengths'].append(chains[chain_id]['lengths'][round_ind])
            img_dict[image_ind][round_ind]['game_ids'].append(game_id)
            img_dict[image_ind][round_ind]['chain_ids'].append(chain_id)
            img_dict[image_ind][round_ind]['chain_ids_shuf'].append(chain_id)

    return img_dict


def pert_sanity_check(test_chains_exp='data/test_shuffle_chains.json', test_segm_exp='data/test_shuffle_segments.json',
                      test_chains='data/test_chains.json', test_segm='data/test_segments.json', n=3):
    """
    Prints the first three segments/chains from the original test set and the experiment set
    """
    print('Sanity check')
    with open(test_chains_exp) as json_file:
        data_chain_shuf = json.load(json_file)
    with open(test_chains) as json_file:
        data_chain = json.load(json_file)


    with open(test_segm_exp) as json_file:
        data_seg_shuf = json.load(json_file)
    with open(test_segm) as json_file:
        data_seg = json.load(json_file)

    for i in range(n):
        print('original chains')
        print(data_chain[i])
        print('experiment chains')
        print(data_chain_shuf[i])
        print()
        print('original segments')
        print(data_seg[i])
        print('experiment segments')
        print(data_seg_shuf[i])
        print("-----------------------------------")


def check_lengths(changed_seg_ids, conditions_inds):
    """Return a dictionary of the number of segments that were changed per condition"""
    lengths = {'hT_nhT':0, 'hT_nhF':0,'hF_nhT':0,'hF_nhF':0,'only_h':0,'only_nh':0,'nothing':0, 'all':0, 'h_nh_all':0}
    for seg_id in changed_seg_ids:
        for condition in conditions_inds:
            if seg_id in conditions_inds[condition]:
                lengths[condition]+=1
                break
    return lengths