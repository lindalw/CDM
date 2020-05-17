import copy
import pandas as pd
import json

def get_seg_ids():
    # pass
    # Load in the segment ids that tell us which segment belongs to which index in the history dataset
    with open('segment_ids_test.json') as json_file:
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

def reorder_datast(dataset_pred_hist):
    # Create new history dataset with the segments in the same order as the no-history dataset
    dataset_pred_hist_cp = copy.deepcopy(dataset_pred_hist)
    # For each segment_id replace the hist_cp with the corresponding data from dataset_pred_hist
    for i in range(len(inv_list)):
        # Replace each key (cause you can't replace entire dicts here apparently...)
        for key in list(dataset_pred_hist[0]):
            dataset_pred_hist_cp[i][key] = dataset_pred_hist[inv_list[i]][key]

    return dataset_pred_hist_cp

get_seg_ids()
