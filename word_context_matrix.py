from collections import Counter
import numpy as np
from scipy import sparse
from options import *

# Build word context dictionary
def build_skipgrams(corpus,token2index):
    skipgram_counts = Counter()
    for doc_index, doc in enumerate(corpus):
        # Build a list of words indices for documents in accordance with сorpus vocabulary
        tokens = [token2index[token] for token in doc]
        for word_index, word in enumerate(tokens):
            left_border_index = max(0, word_index - CONTEXT_WIDTH)
            right_border_index = min(len(doc) - 1, word_index + CONTEXT_WIDTH)
            context_indices = [
                context_index for context_index in range(left_border_index, right_border_index + 1) if context_index != word_index]
            for current_index in context_indices:
                skipgram = (tokens[word_index], tokens[current_index])
                skipgram_counts[skipgram] += 1
    return skipgram_counts

# Build word context matrix
def build_word_context_matrix(skipgrams):
    # Build word context dictionary
    row_index_list = [] # indices of words
    col_index_list = [] # indices of context words
    values = []
    for (word, context), sg_count in skipgrams.items():
        row_index_list.append(word)
        col_index_list.append(context)
        values.append(sg_count)
    # Build word context matrix using csr-matrix
    word_context_matrix = sparse.csr_matrix((values, (row_index_list, col_index_list)))
    return word_context_matrix


# Build SPPMI (smoothed positive pointwise mutual information) matrix
def build_sppmi_matrix(skipgrams_amount, sum_over_words, sum_over_contexts, skipgrams):
    sum_over_words_alpha = sum_over_words ** ALPHA
    whole_sum_over_words_alpha = np.sum(sum_over_words_alpha)

    row_index_list = [] # indices of words
    col_index_list = [] # indices of context words
    sppmi_values = []  # smoothed positive pointwise mutual information

    for (word, context), sg_count in skipgrams.items():
        P_wc = sg_count / skipgrams_amount
        word_sg_amount = sum_over_contexts[word]
        P_w = word_sg_amount / skipgrams_amount

        context_sg_amount_alpha = sum_over_words_alpha[context]
        P_c_alpha = context_sg_amount_alpha / whole_sum_over_words_alpha

        spmi = np.log2(P_wc / (P_w * P_c_alpha))
        sppmi = max(spmi, 0)

        row_index_list.append(word)
        col_index_list.append(context)
        sppmi_values.append(sppmi)

    sppmi_matrix = sparse.csr_matrix((sppmi_values, (row_index_list, col_index_list)))
    return sppmi_matrix
