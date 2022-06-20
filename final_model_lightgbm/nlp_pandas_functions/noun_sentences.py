from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer

from textblob import TextBlob
from collections import Counter
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import tensorflow_hub as hub
import numpy as np
from scipy import linalg as la
from collections import defaultdict
import time

#------------------------


def df_column_noun_sentences_no_embedding(df, column_name, model_error_column, ListingPrice, Transprice, avmvalue):
    noun_sentences_dict = defaultdict(int) 
    noun_sentences_dict_avmerror = defaultdict(int) 
    noun_sentences_dict_avmerror_list = defaultdict(int)    
    noun_sentences_dict_listingerror = defaultdict(int) 
    noun_sentences_dict_listingerror_list = defaultdict(int) 
    noun_sentences_dict_listingid_list = defaultdict(int) 
    
    for iit, text in enumerate(df[column_name].values):
        
        if (iit%1000==0):print(iit, len(df[column_name]))
        
        #tic = time.clock()
        out = noun_sentences_analysis3_no_embedding(text)
        #print(out)
        for phrase in out:
            
            noun_sentences_dict[phrase] += 1
            noun_sentences_dict_avmerror[phrase] += df[model_error_column].values[iit]
            noun_sentences_dict_listingerror[phrase] += (df[Transprice].values[iit]-(df[ListingPrice].values[iit]))\
                                                    /df[Transprice].values[iit]
            
            temp = noun_sentences_dict_avmerror_list[phrase]
            #print(temp)
            #print((temp == 0) | (temp == None))
            if ((temp == 0) | (temp == None)):
                #print(noun_sentences_dict_avmerror_list)
                #print(phrase)
                listtemp = [df[model_error_column].values[iit]]
                noun_sentences_dict_avmerror_list[phrase] = listtemp
                listtemp = [(df[Transprice].values[iit]-(df[ListingPrice].values[iit]))\
                                                    /df[Transprice].values[iit]]
                noun_sentences_dict_listingerror_list[phrase] = listtemp                
                listtemp = [df['listingid'].values[iit]]
                noun_sentences_dict_listingid_list[phrase] = listtemp               
            else:
                #print(noun_sentences_dict_avmerror_list)
                #print(phrase)
                #print(noun_sentences_dict_avmerror_list[phrase])
                noun_sentences_dict_avmerror_list[phrase].append(df[model_error_column].values[iit])
                noun_sentences_dict_listingerror_list[phrase].append((df[Transprice].values[iit]-(df[ListingPrice].values[iit]))\
                                                    /df[Transprice].values[iit])                
                noun_sentences_dict_listingid_list[phrase].append(df['listingid'].values[iit])            
            
        #print(time.clock()-tic)
    
    #ngram_df = pd.DataFrame(sorted(ngram_dict.items(),key=lambda x:x[1],reverse=True))
    return noun_sentences_dict, noun_sentences_dict_avmerror, noun_sentences_dict_avmerror_list, noun_sentences_dict_listingerror, noun_sentences_dict_listingerror_list, noun_sentences_dict_listingid_list


def noun_sentences_analysis3_no_embedding(strtext):

    textblob_temp = TextBlob(strtext)   
    return textblob_temp.np_counts



def df_column_noun_sentences(df, column_name, model_error_column, ListingPrice, Transprice, avmvalue):
    noun_sentences_dict = defaultdict(int) 
    noun_sentences_dict_avmerror = defaultdict(int) 
    noun_sentences_dict_listingerror = defaultdict(int) 
    
    average_embedding = np.zeros([512, len(df[column_name])])
    average_embedding_amplitute = np.zeros([512, len(df[column_name])])
    average_embedding_scaled = np.zeros([512, len(df[column_name])])
    
    for iit, text in enumerate(df[column_name].values):
        
        #print(iit, len(df[column_name]), df[model_error_column].values[iit],
        #     (df[Transprice].values[iit]-(df[avmvalue].values[iit]))/df[Transprice].values[iit],
        #     (df[Transprice].values[iit]-(df[ListingPrice].values[iit]))/df[Transprice].values[iit])
        
        #tic = time.clock()
        out = noun_sentences_sentiment_analysis3(text)
        #print(out)
        for phrase in out[0]:
            noun_sentences_dict[phrase] += 1
            noun_sentences_dict_avmerror[phrase] += df[model_error_column].values[iit]
            noun_sentences_dict_listingerror[phrase] += (df[Transprice].values[iit]-(df[ListingPrice].values[iit]))\
                                                    /df[Transprice].values[iit]
            
        average_embedding[:, iit] = out[3]
        average_embedding_amplitute[:, iit] = out[1]
        average_embedding_scaled[:, iit] = out[2]
        
        #print(time.clock()-tic)
    
    #ngram_df = pd.DataFrame(sorted(ngram_dict.items(),key=lambda x:x[1],reverse=True))
    return noun_sentences_dict, noun_sentences_dict_avmerror, noun_sentences_dict_listingerror, average_embedding, average_embedding_amplitute, average_embedding_scaled



def noun_sentences_sentiment_analysis3(strtext):
    analyzer = VaderSentimentIntensityAnalyzer()  
   
    textblob_temp = TextBlob(strtext)
    #print(textblob_temp.sentiment)
    
    #print(textblob_temp.np_counts)
    #print(type(textblob_temp.np_counts))
    
    average_embedding_amplitute = np.zeros(512)
    average_embedding = np.zeros(512)
    average_embedding_scaled = np.zeros(512)
    #print(average_embedding_amplitute.shape)
    
    
    pos = 0
    neu = 0
    
    
    for idd, key in enumerate(textblob_temp.np_counts):
        #print('')
        #print(key)
        #print(type(key))
        #tb_key = TextBlob(key)
        #print(analyzer.polarity_scores(key))
        #print(analyzer.polarity_scores(key)['pos'])
        #print(analyzer.polarity_scores(key).values)
        #print(tb_key.sentiment)
        #print(list(tb_key.sentiment))
        embedding = embed([key]).numpy()
        average_embedding_amplitute += np.abs(embedding.reshape(512))
        average_embedding += embedding.reshape(512)
        average_embedding_scaled += embedding.reshape(512) * analyzer.polarity_scores(key)['pos']
        
        #print(embedding)
        #print(np.linalg.norm(embedding))
    
    average_embedding_amplitute =  average_embedding_amplitute/len(textblob_temp.np_counts)
    average_embedding =  average_embedding/len(textblob_temp.np_counts)
    average_embedding_scaled = average_embedding_scaled/len(textblob_temp.np_counts)
    
    return textblob_temp.np_counts, average_embedding_amplitute, average_embedding_scaled, average_embedding



def noun_sentences_sentiment_analysis3(strtext):
    analyzer = VaderSentimentIntensityAnalyzer()  
   
    textblob_temp = TextBlob(strtext)
    #print(textblob_temp.sentiment)
    
    #print(textblob_temp.np_counts)
    #print(type(textblob_temp.np_counts))
    
    average_embedding_amplitute = np.zeros(512)
    average_embedding = np.zeros(512)
    average_embedding_scaled = np.zeros(512)
    #print(average_embedding_amplitute.shape)
    
    
    pos = 0
    neu = 0
    
    
    for idd, key in enumerate(textblob_temp.np_counts):
        #print('')
        #print(key)
        #print(type(key))
        #tb_key = TextBlob(key)
        #print(analyzer.polarity_scores(key))
        #print(analyzer.polarity_scores(key)['pos'])
        #print(analyzer.polarity_scores(key).values)
        #print(tb_key.sentiment)
        #print(list(tb_key.sentiment))
        embedding = embed([key]).numpy()
        average_embedding_amplitute += np.abs(embedding.reshape(512))
        average_embedding += embedding.reshape(512)
        average_embedding_scaled += embedding.reshape(512) * analyzer.polarity_scores(key)['pos']
        
        #print(embedding)
        #print(np.linalg.norm(embedding))
    
    average_embedding_amplitute =  average_embedding_amplitute/len(textblob_temp.np_counts)
    average_embedding =  average_embedding/len(textblob_temp.np_counts)
    average_embedding_scaled = average_embedding_scaled/len(textblob_temp.np_counts)
    
    return textblob_temp.np_counts, average_embedding_amplitute, average_embedding_scaled, average_embedding


def noun_sentences_sentiment_analysis2(strtext, ngram, ngram_dict):

    ngram_list = generate_n_grams(strtext, ngram)
    for word in ngram_list:
        ngram_dict[word] += 1
   
    return

#------------------------


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def pca_analysis(array):
    cov_1 = np.dot (array.T, array) 
    print(cov_1.shape)
    cov_2 = np.dot (array, array.T)
    print(cov_2.shape)
    e_vals_1, e_vecs_1 = la.eig (cov_1)
    sum_evals_1 = np.sum(e_vals_1)
    var_exp_1 = (e_vals_1 / sum_evals_1) * 100
    
    e_vals_2, e_vecs_2 = la.eig (cov_2)
    sum_evals_2 = np.sum(e_vals_2)
    var_exp_2 = (e_vals_2 / sum_evals_2) * 100    
    
    return e_vecs_1, var_exp_1, e_vecs_2, var_exp_2


def pca_analysis(array):
    cov_1 = np.dot (array.T, array) 
    print(cov_1.shape)
    cov_2 = np.dot (array, array.T)
    print(cov_2.shape)
    e_vals_1, e_vecs_1 = la.eig (cov_1)
    sum_evals_1 = np.sum(e_vals_1)
    var_exp_1 = (e_vals_1 / sum_evals_1) * 100
    
    e_vals_2, e_vecs_2 = la.eig (cov_2)
    sum_evals_2 = np.sum(e_vals_2)
    var_exp_2 = (e_vals_2 / sum_evals_2) * 100    
    
    return e_vecs_1, var_exp_1, e_vecs_2, var_exp_2

def embed(input):
    return model(input)


def noun_sentences_sentiment_analysis(strtext):
    analyzer = VaderSentimentIntensityAnalyzer()  
   
    textblob_temp = TextBlob(strtext)
    #print(textblob_temp.sentiment)
    
    
    #print(textblob_temp.np_counts)
    #print(type(textblob_temp.np_counts))
    
    embedding_array = np.zeros([512, len(textblob_temp.np_counts)]) * np.nan
    average_embedding_amplitute = np.zeros(512)
    average_embedding = np.zeros(512)
    average_embedding_scaled = np.zeros(512)
    print(average_embedding_amplitute.shape)
    
    for idd, key in enumerate(textblob_temp.np_counts):
        print('')
        print(key)
        print(type(key))
        tb_key = TextBlob(key)
        print(analyzer.polarity_scores(key))
        print(analyzer.polarity_scores(key)['pos'])
        print(tb_key.sentiment)
        embedding = embed([key]).numpy()
        embedding_array[:, idd] = embedding
        average_embedding_amplitute += np.abs(embedding.reshape(512))
        average_embedding += embedding.reshape(512)
        average_embedding_scaled += embedding.reshape(512) * analyzer.polarity_scores(key)['pos']
        
        #print(embedding)
        #print(np.linalg.norm(embedding))
    
    average_embedding_amplitute /=   len(textblob_temp.np_counts)
    e_vecs_1, var_exp_1, e_vecs_2, var_exp_2 = pca_analysis(embedding_array)
             
    return textblob_temp.np_counts, average_embedding_amplitute, average_embedding_scaled, average_embedding,  e_vecs_1, var_exp_1, e_vecs_2, var_exp_2