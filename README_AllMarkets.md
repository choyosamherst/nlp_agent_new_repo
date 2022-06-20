# nlp_agent


Notebook workflow order:

----------

1.  nlp_clean_preprocess_sentiment_workflow_all_markets.ipynb:

Basic cleaning and preparation of the pandas dataframe 
Filtered by df_sales['ListingPrice']/df_sales['Transprice']  and
df_sales['amvValue']/df_sales['Transprice']  0.4 and 1.5

Also includes nltk sentiment analysis.

Last run: 2022/04/09

output: agent_comments_XX_filtered.fea

----------

2. sentence_composition_workflow_all_markets.ipynb

Last run: 2022/04/09 (RUNNING SLOW)

output: agent_comments_XX_sentence_filtered_composition.fea


----------

3. embeddings_workflow_all_markets.ipynb

Last run:  2022/04/09 (RUNNING SLOW)

output:agent_comments_XX_sentence_filtered_ _embedding.fea


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


------------

noun_sentences_workflow_noembedding_all_markets

noun_sentence_filtering_all_markets


noun_sentence_final_set_stats_and_subset_for_prediction_all_markets

