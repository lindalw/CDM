import numpy as np
import json

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