import sklearn.feature_extraction.text as text
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk import ngrams


#------------------------

def get_n_grams_probability(strtext, ngram):
    
    return


#------------------------


def df_lambda_ngram(df, column_name, ngram):
    ngram_dict = defaultdict(int)    
    df[column_name].apply(lambda x: df_get_n_grams_count(x, ngram, ngram_dict))
    ngram_df = pd.DataFrame(sorted(ngram_dict.items(),key=lambda x:x[1],reverse=True))
    return ngram_df

def df_get_n_grams_count(strtext, ngram, ngram_dict):

    ngram_list = generate_n_grams(strtext, ngram)
    for word in ngram_list:
        ngram_dict[word] += 1
   
    return

#https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/
def generate_n_grams(strtext, ngram):
    #words=[word for word in text.split(" ") if word not in set(stopwords.words('english'))]  
    #print("Sentence after removing stopwords:",words)
    words=[word for word in strtext.split()]
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans


def count_n_grams(ngram_list):
    ngram_dict = defaultdict(int)  
    for word in ngram_list:
        ngram_dict[word] += 1

    return ngram_dict


def get_n_grams_count(strtext,ngram):
    
    #print(strtext)
    ngram_list = generate_n_grams(strtext, ngram)
    #print('ngram list: ',ngram_list)
    ngram_dict = count_n_grams(ngram_list)
    ngram_df = pd.DataFrame(sorted(ngram_dict.items(),key=lambda x:x[1],reverse=True))
    
    return ngram_df


def get_n_grams_count(strtext,ngram):
    
    ngram_list = generate_n_grams(strtext, ngram)
    ngram_dict = count_n_grams(ngram_list)
    ngram_df = pd.DataFrame(sorted(ngram_dict.items(),key=lambda x:x[1],reverse=True))
    
    return ngram_df



#-----------------------
#https://importsem.com/build-an-n-gram-text-analyzer-for-seo-using-python/
#NO API KEY yet
    
def kg(keywords):

    kg_entities = []
    apkikey=''

    for x in keywords:
        url = f'https://kgsearch.googleapis.com/v1/entities:search?query={x}&key={apikey}&limit=1&indent=True'
        payload = {}
        headers= {}
        response = requests.request("GET", url, headers=headers, data = payload)
        data = json.loads(response.text)

    try:
        getlabel = data['itemListElement'][0]['result']['@type']
        score = round(float(data['itemListElement'][0]['resultScore']))
    except:
        score = 0
        getlabel = ['none']

    labels = ""

    for item in getlabel:
        labels += item + ","
    labels = labels[:-1].replace(",",", ")

    if labels != ['none'] and score > 500:
        kg_subset = []
        kg_subset.append(x)
        kg_subset.append(score)
        kg_subset.append(labels)

        kg_entities.append(kg_subset)
    return kg_entities



#-----------------------
#Not using this function...
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return matrix,pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values

#------------------------
#Not using this function...

def bigrams_join(n):
    index=0
    for doc in all_docs: 
        bigrams = list(ngrams(doc , n))
        for i in bigrams:
            words = " ".join(i)
            yield [words,df['avmerror'][index],abs(df['avmerror'][index])]
        index+=1
        if index % 10000 == 0: logging.info("Processed %d listings" %index)