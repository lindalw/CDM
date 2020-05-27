import numpy as np
import json
from helpers import *
import random

def chain_shuffle(chain_copy, chain_file='data/test_shuffle_chains.json', segment_file='data/test_shuffle_segments.json', output=True):
    """
    Shuffle the segments in a chain
    
    Chain_copy is a deepcopy of chain_test_set.chains
    """
    # Iterate over the indices in the chain set
    for sample_ind in range(len(chain_copy)):
        # Shuffle the orders of the segment in the chain
        inds = list(range(len(chain_copy[sample_ind]['segments'])))
        np.random.shuffle(inds)
        chain_copy[sample_ind]['segments'] = [chain_copy[sample_ind]['segments'][i] for i in inds]
        chain_copy[sample_ind]['lengths'] = [chain_copy[sample_ind]['lengths'][i] for i in inds]
    
    # Write new chains file
    if output:
        with open(chain_file, 'w') as json_file:
            json.dump(chain_copy, json_file)
        # Keep segment file the same
        with open('data/test_segments.json') as json_file:
            segments = json.load(json_file)
        with open(segment_file, 'w') as json_file:
            json.dump(segments, json_file)
    
    return chain_copy, segments


# def pert_exchange_games(chain_copy, round_id, chain_file_start='data/test_games_', segment_file_start='data/test_games_', output=True):
#     """ 
#     Chain_copy is a deepcopy of chain_test_set.chains
#     Changes the segements in round round_id for another segment on the same image another game (including its target). 
#     Returns the new chain, segment and list of the segment ids that were changed
#     """
    
#     chain_test_game_n, changed_seg_ids = exchange_games(chain_copy, round_id)
    
#     # Write new chains file
#     if output:
#         with open(chain_file_start+str(round_id)+'_chains.json', 'w') as json_file:
#             json.dump(chain_test_game_n, json_file)
#         # Keep segment file the same
#         with open('data/test_segments.json') as json_file:
#             segments = json.load(json_file)
#         with open(segment_file_start+str(round_id)+'_segments.json', 'w') as json_file:
#             json.dump(segments, json_file)
#     return chain_test_game_n, segments, changed_seg_ids


# def exchange_games(chains, round_id):
#     """
#     chains is chain_test_set.chains
#     round_id is the round that we want to change

#     Changes the segements in round round_id for another segment on the same image another game (including its target). 
    
#     Returns the new chain test set and the segment indices of the new segments in the chains.
#     """
#     # Get a deepcopy of the chain list  
#     chain_test_game_n = copy.deepcopy(chains)   
#     img_dict = get_img_dict(chains)
#     changed_seg_ids = []
    
#     # Shuffle the chain indices so that each segment still occurs once
#     for img_id in img_dict:
#         if round_id in img_dict[img_id]:
#             random.shuffle(img_dict[img_id][round_id]['chain_ids_shuf'])

#     # Iterate over the indices in the chain set
#     for chain_id in range(len(chain_test_game_n)):
#         img_id = chain_test_game_n[chain_id]['target']
#         # game_id_cur = chain_test_game_n[chain_id]['game_id']

#         # Check if there are at least round_id rounds in this chain
#         if len(chain_test_game_n[chain_id]['segments']) <= round_id:
#             continue
#         # Check if there are other games
#         if len(set(img_dict[img_id][round_id]['game_ids'])) == 1:
#             continue

#         # Get index of the new segment, given the shuffled chain ids
#         new_seg_ind = img_dict[img_id][round_id]['chain_ids_shuf'].index(chain_id)

#         # Change segment in round_id to the new_seg_id
#         chain_test_game_n[chain_id]['segments'][round_id] = img_dict[img_id][round_id]['segments'][new_seg_ind]
#         chain_test_game_n[chain_id]['lengths'][round_id] = img_dict[img_id][round_id]['lengths'][new_seg_ind]

#         # Save new segment id for analysis
#         changed_seg_ids.append(img_dict[img_id][round_id]['segments'][new_seg_ind])
#     return chain_test_game_n, changed_seg_ids



# def pert_exchange_imgs(chain_copy, round_id, chain_file_start='data/test_imgs_', segment_file_start='data/test_imgs_', output=True):
#     """ 
#     Chain_copy is a deepcopy of chain_test_set.chains
#     """
    
#     chain_test_game_n, changed_seg_ids = exchange_images(chain_copy, round_id)
    
#     # Write new chains file
#     if output:
#         with open(chain_file_start+str(round_id)+'_chains.json', 'w') as json_file:
#             json.dump(chain_test_game_n, json_file)
#         # Keep segment file the same
#         with open('data/test_segments.json') as json_file:
#             segments = json.load(json_file)
#         with open(segment_file_start+str(round_id)+'_segments.json', 'w') as json_file:
#             json.dump(segments, json_file)
#     return chain_test_game_n, segments, changed_seg_ids


# def exchange_images(chains, round_id):
#     """
#     chains is chain_test_set.chains
#     round_id is the round that we want to change

#     Changes the segements in round round_id for another segment of the same game but another image (including its target). 
#     Returns the new chain test set and the segment indices of the new segments in the chains.
#     """
#     # Get a deepcopy of the chain list
#     chain_test_game_n = copy.deepcopy(chains)   
#     game_dict = get_game_dict(chains)
#     changed_seg_ids = []
    
#     # Shuffle the target indices
#     for game_id in game_dict:
#         if round_id in game_dict[game_id]:
#             random.shuffle(game_dict[game_id][round_id]['chain_ids_shuf'])

#     # Iterate over the indices in the chain set
#     for chain_id in range(len(chain_test_game_n)):
#         game_id = chain_test_game_n[chain_id]['game_id']

#         # Check if there are at least round_id rounds in this chain
#         if len(chain_test_game_n[chain_id]['segments']) <= round_id:
#             continue
#         # Check if there are other games
#         if len(set(game_dict[game_id][round_id]['targets'])) == 1:
#             continue

#         # Get index of the new segment, given the shuffled chain ids
#         new_seg_ind = game_dict[game_id][round_id]['chain_ids_shuf'].index(chain_id)

#         # Change segment in round_id to the new_seg_id
#         chain_test_game_n[chain_id]['segments'][round_id] = game_dict[game_id][round_id]['segments'][new_seg_ind]
#         chain_test_game_n[chain_id]['lengths'][round_id] = game_dict[game_id][round_id]['lengths'][new_seg_ind]

#         # Save new segment id for analysis
#         changed_seg_ids.append(game_dict[game_id][round_id]['segments'][new_seg_ind])
#     return chain_test_game_n, changed_seg_ids


# def get_game_dict(chains):
#     """
#     chains is chain_test_set.chains
#     Return a dictionary of
#     {game_id:{'round_id':'segments':[segids],
#                         'lengths':[seglengths],
#                         'game_ids':[gameids]}}
#     where game_id is the game_id of the chain
#     round_id is the round/rank, i.e. the i'th time the image is being talked about
#     segid is the segment id
#     seglength is the length of the segment with the corresponding index
#     gameid is the gameid to which the segment with the corresponding index belongs

#     Used for exchange_images()
#     """
#     game_dict = {}
#     # Get segments, lengths and gameids
#     for chain_id in range(len(chains)):
#         image_ind = chains[chain_id]['target']
#         game_id = chains[chain_id]['game_id']
#         # Add gameid to the game_dict    
#         if game_id not in game_dict:
#             game_dict[game_id] = {}

#         # Iterate over the rounds/segments
#         for round_ind in range(len(chains[chain_id]['segments'])):

#             # Add round to the dict for this target image
#             if round_ind not in game_dict[game_id]:
#                 game_dict[game_id][round_ind] = {'segments':[],
#                                                   'lengths':[],
#                                                  'game_ids':[],
#                                                  'targets':[],
#                                                  'chain_ids':[],
#                                                  'chain_ids_shuf':[]}

#             # Add segment with the length and gameid info to this round dict
#             game_dict[game_id][round_ind]['segments'].append(chains[chain_id]['segments'][round_ind])
#             game_dict[game_id][round_ind]['lengths'].append(chains[chain_id]['lengths'][round_ind])
#             game_dict[game_id][round_ind]['game_ids'].append(game_id)
#             game_dict[game_id][round_ind]['targets'].append(image_ind)
#             game_dict[game_id][round_ind]['chain_ids'].append(chain_id)
#             game_dict[game_id][round_ind]['chain_ids_shuf'].append(chain_id)

#     return game_dict



