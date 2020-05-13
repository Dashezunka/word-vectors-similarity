# Algorithm for similar words searching based on word vectors
It builds a model to search for similar words based on word context matrix and using following similarity metrics:
1. Cosine similarity.
2. Kullback Leibler divergence.
3. Jensen Shannon divergence.

##Description
The algorithm works in the following way:
- Builds tokenized and lemmatized corpus removing stop words. 
- Builds word context dictionary for pre-processed corpus.
- Forms corpus vocabulary.
- Builds word context dictionary based on corpus vocabulary.
- Build word context matrix using csr-matrix.
- Build SPPMI (smoothed positive pointwise mutual information) matrix.
- Searches top N similar words for chosen one using one of described similarity metrics.
- Makes a comparison of model results and *gold standard* values for words from `wordsim353` corpus.
- Saves results in csv-files.

**Note:** Special dicts `token2index` and `index2token` are built to optimize memory usage.

##Install and configure
You need to install dependencies from `requirements.txt` using
`pip3 install -r requirements.txt`   

Adjust further params in `options.py` before starting the model.  
####Execution mode
To adjust **MODE** param choose:
- *"SEARCH_SIMILARS"* if you want to find top N similar words for your one;
 **Note:** You should adjust following params in this case:
  - *TOP_N* is an amount of similar words;
  - *WORD* is a word for which similar words are searched;
- *"WORDSIM_ANALYSIS"* if you want to compare model results with gold standard values for *wordsim353 words*.

**Note:** Set *CORPUS_PATH* according to chosen execution mode.

####Similarity metrics
To adjust **METRIC** choose one of the following similarity metrics:
1. Cosine similarity.
2. Kullback Leibler divergence.
3. Jensen Shannon divergence.

**Note:** If you want to use your own corpus, set the CATEGORY_PATH in `options.py`.  
You can also adjust *CONTEXT_WIDTH* for word context matrix and *alpha-param* for SPPMI-matrix there. 

##Running command
Try `python3 word_embeddings.py` in the project directory.