CONTEXT_WIDTH = 5
ALPHA = 0.75 # param for SPPMI-matrix

# Execution mode:
# choose "SEARCH_SIMILARS" if you want to find top N similar words for your one or
# choose "WORDSIM_ANALYSIS" if you want to compare model results with gold standard values for wordsim words
MODE = "SEARCH_SIMILARS" # "WORDSIM_ANALYSIS"

# Adjust corpus path according to chosen execution mode
# ("data/training_corpus/" or "data/wordsim_corpus/"):
CORPUS_PATH = "data/phones_corpus/"

# Choose one of the following similarity metrics:
# 1 - cosine similarity
# 2 - Kullback_Leibler_divergence
# 3 - Jensen_Shannon_divergence
METRIC = '3'

# In case of "SEARCH_SIMILARS" mode adjust these params:
TOP_N = 10 # amount of similar words
WORD = "ios" # a word for which similar words are searched