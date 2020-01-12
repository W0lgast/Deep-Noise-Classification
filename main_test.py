"""
Loads a pickled model, then does stuff to it I guess.

Kipp McAdam Freud
12/01/2019
"""
# --------------------------------------------------------------

import pickle

import util.utilities as ut

# --------------------------------------------------------------

FILE_PATH = "models/four_fold_cnn.pkl"

# --------------------------------------------------------------

infile = open(FILE_PATH,'rb')
model = pickle.load(infile)
infile.close()

ut.exit(1)