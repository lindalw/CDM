# CDM

Photobook dataset: investigating the importance of history in a reference resolution model
Bob Borsboom (10802975), Emma Hokken (10572090), Linda Wouters (11000139)
CDM Group 3

This github page contains the code corresponding to the report for the Computational Dialogue Modeling course given by Raquel Fernandez Rovira.

The project is based on the data and pre-trained model from Haber et al. (2019; https://dmg-photobook.github.io/). 

The code in this github uses the code from: https://github.com/dmg-photobook/photobook_dataset/tree/master/discriminator.

The pretrained models model_history_blind_accs_2019-02-20-14-22-23.pkl, model_blind_accs_2019-02-17-21-18-7.pkl and model_history_noimg_accs_2019-03-01-14-25-34.pkl were received from https://github.com/dmg-photobook/photobook_acl_models.

The following files in the photobook_dataset/discriminator folder are created by us:

The notebooks contain the results for the report:
* Experiments_run_and_analyses.ipynb - this loads the pretrained models, creates the perturbation experiment files, runs these experiments and analyses the results. The experiment files are created so that the start of the segment and chain file names begin with the experiment name (e.g. 'test_games_0_segments.json). This way the split (e.g. 'test_games_0') can be given as input to the get_predictions() function in get_predictions.py.

* Cosine_similarity.ipynb - contains the cosine similarity analysis on the dataset

* Pos_tag_distribution.ipynb - contains the POS tag distribution analysis on the dataset

Additionally:
- helpers.py - contains the functions needed to get the predictions from the models, seperate the predictions into different conditions (where the segments are correctly or incorrectly classified) and to calculate their accuracies, and to get the relevant data for the analyses.

- get_predictions.py - contains the functions to get the predictions of the model for the segment and chain files that start with a specific split (e.g. 'test' or 'test_games_0').

- perturbations.py - contains the function that shuffles the order of the segments in the chain (other perturbation functions are in the Experiments_run_and_analyses.ipynb)

- pos_tag_distribution_helpers.py - contains all the functions which were needed to perform the pos tag experiments and analysis, i.e.: removing pos tags from the segments and analysing the ratio in pos tags between different ranks.

- in train_nohistory.py and train_history.py the get_predictions functions are altered to add the predictions to the dataset


References:

Janosch Haber, Tim Baumgartner, Ece Takmaz, Lieke Gelderloos, Elia Bruni, and Raquel Fernandez, 2019. The photobook dataset: Building  common ground through visually-grounded dialogue.
