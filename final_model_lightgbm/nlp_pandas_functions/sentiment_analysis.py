from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer

from textblob import TextBlob

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#nltk.download([
#     "names",
#     "stopwords",
#     "state_union",
#     "twitter_samples",
#     "movie_reviews",
#     "averaged_perceptron_tagger",
#     "vader_lexicon",
#     "punkt",
# ])

import flair
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

from sparknlp.pretrained import PretrainedPipeline


#----------------

def single_entry_nltk_polarity_score(strtext):
    
    sia = SentimentIntensityAnalyzer()    

    sentiment =  sia.polarity_scores(strtext)
    
    return sentiment


def single_entry_vader_polarity(strtext):
    
    sia = VaderSentimentIntensityAnalyzer()    

    sentiment =  sia.polarity_scores(strtext)
    
    return sentiment


def single_entry_textblob_polarity(strtext): 

    sentiment =  TextBlob(strtext).sentiment
    
    return sentiment


#----------------


def df_lambda_nltk_polarity_score(df, columns):
    sia = SentimentIntensityAnalyzer()    
    for column in columns:
        new_column = column + '_polarity'
        df[new_column] = df[column].apply(lambda x: 
                           list(sia.polarity_scores(x).values()))
    
    return df


def df_lambda_nltk_polarity_score_dict(df, columns):
    sia = SentimentIntensityAnalyzer()    
    for column in columns:
        new_column = column + '_nltkpolarity'
        df[new_column] = df[column].apply(lambda x: 
                           sia.polarity_scores(x))
    
    return df


def df_lambda_textblob_polarity(df, columns):
      
    for column in columns:
        new_column = column + '_textblobpolarity'
        df[new_column] = df[column].apply(lambda x: TextBlob(x).sentiment)
    
    return df


def df_lambda_vader_polarity(df, columns):
    analyzer = VaderSentimentIntensityAnalyzer()  
    for column in columns:
        new_column = column + '_vaderpolarity'
        df[new_column] = df[column].apply(lambda x: analyzer.polarity_scores(x))
    
    return df


def df_lambda_flair_polarity(df, columns):
      
    for column in columns:
        new_column = column + '_flairpolarity'
        df[new_column] = df[column].apply(lambda x: flair_steps(x))
    
    return df

def flair_steps(strtext):
    s = flair.data.Sentence(strtext)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    return total_sentiment

def df_lambda_list(df, columns):
      
    for column in columns:
        new_column = column + '_list'
        df[new_column] = df[column].apply(lambda x: list(x))
    
    return df

