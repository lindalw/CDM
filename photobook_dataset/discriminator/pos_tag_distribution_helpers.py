import gensim.downloader as api
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.decomposition import PCA

import numpy as np
import pickle
import json
import copy
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats


from Vocab import Vocab
vocab = Vocab('data/vocab.csv', 3)
from get_predictions import get_predictions
from helpers import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from ChainDataset import ChainDataset

chain_test_set = ChainDataset(
    data_dir='data/',
    segment_file='segments.json',
    chain_file='test_chains.json',
    vectors_file='vectors.json',
    split='test'
)

from SegmentDataset import SegmentDataset

segment_test_set = SegmentDataset(
    data_dir='data/',
    segment_file='segments.json',
    vectors_file='vectors.json',
    split='test'
)


from collections import Counter
import nltk
import pickle
from nltk import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('universal_tagset')

import matplotlib
import matplotlib.pyplot as plt

with open('oov_dictionary.pickle', 'rb') as f:
    oov_dict = pickle.load(f)

top_10 = ['NOUN', 'DET', 'VERB', 'ADP', 'ADJ', '.', 'ADV', 'NUM', 'PRON', 'CONJ']
top_4 = ['NOUN', 'VERB', 'ADJ', 'ADV']

''' explanation:

ADJ	adjective	new, good, high, special, big, local
ADP	adposition	on, of, at, with, by, into, under
ADV	adverb	really, already, still, early, now
CONJ	conjunction	and, or, but, if, while, although
DET	determiner, article	the, a, some, most, every, no, which
NOUN	noun	year, home, costs, time, Africa
NUM	numeral	twenty-four, fourth, 1991, 14:24
PRT	particle	at, on, out, over per, that, up, with
PRON	pronoun	he, their, her, its, my, I, us
VERB	verb	is, say, told, given, playing, would
.	punctuation marks	. , ; !
X	other	ersatz, esprit, dunno, gr8, univeristy

'''




def pos_tag_distribution_per_group(group, condition_seg_hist, vocab, total_first_seg_dict, total_current_dict):
    print("GOI")
    segments = list(condition_seg_hist[group])
    for segment in tqdm(segments):
        if condition_seg_hist[group][segment] == {}:
            continue

        # Get sentences per segment
        first_seg, current_seg = convert_to_sentences(segment, condition_seg_hist[group])
        print("first_seg", first_seg)
        first_sentence_dict = sentence_to_pos_tags(first_seg)
        print("HIERO", first_sentence_dict)
        current_sentence_dict = sentence_to_pos_tags(current_seg)

        # Update dicts
        update_dict(first_sentence_dict, total_first_seg_dict)
        update_dict(current_sentence_dict, total_current_dict)

    return total_first_seg_dict, total_current_dict


def convert_to_sentences(segment, dataset, vocab):
    # Decode first and current sentence
    first_seg = list(dataset[segment].values())[0]['first_seg']

    current_seg = list(dataset[segment].values())[0]['current_seg']
    dec_first_seg = vocab.decode(first_seg)
    dec_current_seg = vocab.decode(current_seg)

    return dec_first_seg, dec_current_seg


def update_dict(new_dict, total_dict):
    for key, value in new_dict.items():
        new_key = key
        new_value = value

        for key, value in total_dict.items():
            if key == new_key:
                current_value = total_dict[key]
                update_value = current_value + new_value[0]
                total_dict[key] = update_value

    return None


def dict_to_result(first_seg_dict, group, result, place):
    #   convert dict to list for analysis

    dict_list = []
    total = 0
    for key, value in first_seg_dict.items():
        temp = [value, key]
        total += value
        dict_list.append(temp)

    # Convert to percentages
    for tag in dict_list:
        tag[0] = tag[0] / total

    dict_list.sort(reverse=True)
    group = group + "_" + place

    result[group] = dict_list

    return result


def sentence_to_pos_tags(sentence):
    lower_case_sentence = []
    correct_sentence = []
    pos_tags = []

    #   lower the sentences
    for word in sentence:
        lower_case_sentence.append(word.lower())

    #   check if word is misspelled, the oov_dict can maybe corrrect that
    for word in lower_case_sentence:

        if word not in oov_dict:
            correct_sentence.append(word)
        else:
            correct_sentence.append(oov_dict[word])

    # Tag the sentence , so for every sentence got the pos tags
    try:
        pos_tags += [word[1] for word in pos_tag(correct_sentence, tagset='universal')]

    except:
        None

    pos_tag_dict = dict()
    for key, value in dict(Counter(pos_tags)).items():
        pos_tag_dict[key] = [value]

    return pos_tag_dict


# plot the pie charts per group

def plot_dict(plot_title, values):
    labels = []
    all_values = []

    for v in range(len(values)):
        # print("v", values[v])

        labels.append(values[v][1])
        all_values.append(values[v][0])

    fig1, ax1 = plt.subplots()
    ax1.pie(all_values, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    fig1.suptitle(plot_title, fontsize=20)
    plt.show()


def create_history_dict(chain_data):
    history_dict = {}

    # Loop through all segments
    for segment in tqdm(range(0, 6801)):

        # For each segment, find its place in the chain_data and add this to history dict
        for index, chain_data_row in enumerate(chain_data):
            chain_segments = chain_data_row['segments']

            if segment in chain_segments:

                # If segment already exits, add new index
                if segment in history_dict:
                    current_list = history_dict[segment]
                    current_list.append(index)
                    history_dict[segment] = current_list
                # If segment does not yet exist in dict, add it
                else:
                    history_dict[segment] = [index]

    return history_dict

    # geef de segment_id en het returnt de index waar het in staat

    # segment_id: [ index]
    # {0: [0],
    #  7: [0],
    #  8: [0, 4],


def update_history_dataset(segment_dataset, history_dict, chain_data):
    for segment_id, segment_data in enumerate(segment_dataset):
        segment_length = segment_data['length']

        # Find at which index in history file a certain segment is
        index = history_dict[segment_id]

        # Retrieve all indices in history file and change length
        for i in index:
            all_segments_for_that_index = chain_data[i]['segments']
            for ii, seg in enumerate(all_segments_for_that_index):

                if segment_id == seg:
                    chain_data[i]['lengths'][ii] = segment_length

    return chain_data


def import_dialogue_data():
    from ChainDataset import ChainDataset

    chain_test_set = ChainDataset(
        data_dir='data/',
        segment_file='segments.json',
        chain_file='test_chains.json',
        vectors_file='vectors.json',
        split='test'
    )

    from SegmentDataset import SegmentDataset

    segment_test_set = SegmentDataset(
        data_dir='data/',
        segment_file='segments.json',
        vectors_file='vectors.json',
        split='test'
    )

    segment_test_set = copy.deepcopy(segment_test_set)
    chain_data = copy.deepcopy(chain_test_set.chains)

    return segment_test_set, chain_data


def remove_tag(tag, segment_test_set, percentage_remove):
    for segment_id, segment_data in enumerate(segment_test_set):

        pos_tags = []
        lower_case_sentence = []
        correct_sentence = []

        encoded_segment = vocab.decode(segment_data['segment'])

        for word in encoded_segment:
            if word == '-A-':
                lower_case_sentence.append(word)
                continue
            if word == '-B-':
                lower_case_sentence.append(word)
                continue
            lower_case_sentence.append(word.lower())

        for word in lower_case_sentence:

            if word not in oov_dict:
                correct_sentence.append(word)
            else:
                correct_sentence.append(oov_dict[word])

        try:
            pos_tags += [word[1] for word in pos_tag(correct_sentence, tagset='universal')]

            # List with indices to remove
            remove_index = []
            for index, word in enumerate(pos_tags):
                if word == tag:
                    remove_index.append(index)

            #          loop door de candidaten die je wilt verwijderen. met een bepaalde kans worden ze ook echt verwijdert.
            new_remove = []
            for ind in remove_index:
                p = np.random.rand()
                if p < percentage_remove:
                    new_remove.append(ind)
            remove_index = new_remove

            # Only keep certain percentage of indices to remove
            #             remove_index = np.random.choice(remove_index, int(remove_perc*len(remove_index)))

            for index in reversed(remove_index):
                del correct_sentence[index]

            encoded_sentence = vocab.encode(correct_sentence)

            segment_test_set[segment_id]['segment'] = encoded_sentence
            segment_test_set[segment_id]['length'] = len(encoded_sentence)

        except IndexError:
            continue

    return segment_test_set

def dump_json_files(chain_data, segment_data, tag, percentage_remove):

    print(percentage_remove)


    # write chain information to file
    name = 'data/test_' + str(tag) + '_' + str(percentage_remove) +'_chains.json'
    print("name", name)

    with open(name, 'w') as json_file:
        json.dump(chain_data, json_file)


    # SEGMENT SAVE
    name = 'data/test_' + str(tag) + '_' + str(percentage_remove) +'_segments.json'
    print("name", name)

    segment_data_test = [seg for seg in segment_data]
    with open(name, 'w') as json_file:
        json.dump(segment_data_test, json_file)

    print("JSON files are saved :), you're done...")