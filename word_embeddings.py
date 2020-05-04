from os.path import join, isfile
from os import listdir
import csv
from text_preprocessing import *
from word_context_matrix import *
from similarity_metrics import *

# Get list of N most similars for word
def get_top_n_similars(word, matrix, sim_metric, reverse_flag = True):
    if not token2index.get(word):
        print("Error! The word doesn't exist in corpus!")
    word_vector = matrix.getrow(token2index[word]).toarray()[0]
    similar_tokens = []
    for token_index in token2index.values():
        token_vector = matrix.getrow(token_index).toarray()[0]
        similarity = sim_metric(word_vector, token_vector)
        similar_tokens.append((token_index, similarity))
    similar_tokens.sort(key=lambda x: x[1], reverse = reverse_flag)
    return [(index2token[token_index], similarity) for token_index, similarity in  similar_tokens[:TOP_N]]


# Compare model results for wordsim words with gold standard values
def compare_with_wordsim(matrix, sim_metric):
    with open('data/wordsim_similarity_goldstandard.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        result = []
        for row in csv_reader:
            word1 = row[0].lower()
            word2 = row[1].lower()
            wordsim_similarity = row[2]
            # check if both words from wordsim exist in model dictionary
            if token2index.get(word1) and token2index.get(word2):
                v1 = matrix.getrow(token2index[word1]).toarray()[0]
                v2 = matrix.getrow(token2index[word2]).toarray()[0]
                model_similarity = sim_metric(v1, v2)
                result.append((word1, word2, wordsim_similarity, model_similarity))
        return result

# Build tokenized and lemmatized wordsim_corpus without stop words
corpus = []
document_name_list = [f for f in listdir(CORPUS_PATH) if isfile(join(CORPUS_PATH, f))]
for name in document_name_list:
    with open(join(CORPUS_PATH, name), 'r') as doc:
        text = doc.read()
        sentences = tokenize_text(text)
        for sentence in sentences:
            lemmas = lemmatize_text(sentence)
            preprocessed_sentence = remove_stop_words(lemmas)
            if len(preprocessed_sentence) < 2:
                continue
            corpus.append(preprocessed_sentence)

# Form corpus vocabulary
corpus_vocabulary = set()
for document in corpus:
    corpus_vocabulary.update(document)

# Build dicts to optimize memory usage
token2index = {token: index for index, token in enumerate(corpus_vocabulary)}
index2token = {index: token for token, index in token2index.items()}
# print("token2index", token2index)
# print("index2token", index2token)

# Build word context dictionary
skipgram_counts = build_skipgrams(corpus, token2index)
# Build word context matrix using csr-matrix
word_context_matrix = build_word_context_matrix(skipgram_counts)

# Build SPPMI (smoothed positive pointwise mutual information) matrix
skipgrams_amount = word_context_matrix.sum() # sum over all values of matrix
sum_over_words = np.array(word_context_matrix.sum(axis=0)).flatten() # array of sum over words for context
sum_over_contexts = np.array(word_context_matrix.sum(axis=1)).flatten() # array of sum over contexts for word
sppmi_matrix = build_sppmi_matrix(skipgrams_amount, sum_over_words, sum_over_contexts, skipgram_counts)

# Adjust execution mode of the model
sim_metric = None
reverse_flag = True
metric_name = ''
if METRIC == '1':
    sim_metric = calc_cosine_similarity
    metric_name = "cos_sim"
elif METRIC == '2':
    sim_metric = calc_Kullback_Leibler_divergence
    reverse_flag = False
    metric_name = "KL_div"
elif METRIC == '3':
    sim_metric = calc_Jensen_Shannon_divergence
    reverse_flag = False
    metric_name = "JS_div"
else:
    print("Error! Chosen metric doesn't exist!")
    exit(1)

print('Vocab size is: ', len(corpus_vocabulary))
if MODE == "SEARCH_SIMILARS":
    res = get_top_n_similars(WORD, sppmi_matrix, sim_metric, reverse_flag)
    print(res)
    # Save results as csv
    print("Save results as csv")
    # for top N similars search
    with open('data/results/similars/{0}_{1}.csv'.format(WORD, metric_name), 'w') as out:
        sim_words_out = csv.writer(out)
        sim_words_out.writerow(['word', 'sim_value'])
        for word, sim_value in res:
            sim_words_out.writerow([word, sim_value])

elif MODE == "WORDSIM_ANALYSIS":
    res = compare_with_wordsim(sppmi_matrix, sim_metric)
    # Save results as csv
    print("Save results as csv")
    # for wordsim analysis
    with open('data/results/wordsim_analysis_{0}.csv'.format(metric_name), 'w') as out:
        analysis_out = csv.writer(out)
        analysis_out.writerow(['word1', 'word2', 'golden_value', 'model_value'])
        for word1, word2, golden_value, model_value in res:
            analysis_out.writerow([word1, word2, golden_value, model_value])
else:
    print("Error! Chosen mode doesn't exist!")
    exit(1)

