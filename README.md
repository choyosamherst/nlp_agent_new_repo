# nlp_agent


Notebook workflow order:

----------

1.  nlp_clean_preprocess_workflow.ipynb:

Basic cleaning and preparation of the pandas dataframe 
Filtered by df_sales['ListingPrice']/df_sales['Transprice']  and
df_sales['amvValue']/df_sales['Transprice']  0.4 and 1.5

Last run: 2022/02/21

output: _processed.fea 

----------

2. sentiment_analysis_workflow.ipynb

Last run: 2022/02/21

    output: _withsentiment.feax

----------

3. sentence_composition_workflow.ipynb

Last run: 2022/02/22

output: _sentencecomposition.fea


----------

4. embeddings_workflow.ipynb

Last run: 2022/02/22

output: _withembeddings.fea

----------


5. generate_overall_frequency_ngrams.ipynb

Generate overall ngram counts for all the desired columns

Last run: 2022/02/22

output: ngrams/...._words_Ngram.fea 

----------

6. frequency_ngrams_cumulative_error_workflow


----------




----------




n. frequency_ngrams_post_analysys.ipynb

DO NOT RUN AS IS...
NEEDS OPTIMIZING

----------



----------------

Additional processing notebooks:

A. noun_sentence_order_dict.ipynb

Order noun sentence "translation" dictionary.


----------------





nlp_pandas_functions:

Folder with all the functions use to process pandas dataframe with agent comments




