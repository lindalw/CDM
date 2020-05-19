import copy
import pandas as pd
import json
from tqdm import tqdm

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
    conditions_inds = {'hT_nhT':[], 'hT_nhF':[], 'hF_nhT':[], 'hF_nhF':[], 'only_h':[], 'only_nh':[], 'nothing':[]}
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

        elif 'History' in dataframe[ind]:
            conditions_inds['only_h'].append(ind)

        elif 'No history' in dataframe[ind]:
            conditions_inds['only_nh'].append(ind)
        else:
            conditions_inds['nothing'].append(ind)
    return conditions_inds


def add_chains_rounds(dataset_pred_no_hist, dataset_pred_hist_cp, chain_test_set):
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

    return condition_seg_hist

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
    dataset_pred_no_hist, dataset_pred_hist_cp = add_chains_rounds(dataset_pred_no_hist, dataset_pred_hist_cp, chain_test_set)

    dataframe = get_pred_dataframe(dataset_pred_no_hist, dataset_pred_hist_cp)
    conditions_inds = get_conditions_inds(dataframe)

    condition_seg_hist = get_condition_seg_hist(conditions_inds, dataset_pred_hist_cp)

    return dataset_pred_no_hist, dataset_pred_hist_cp, conditions_inds, condition_seg_hist
