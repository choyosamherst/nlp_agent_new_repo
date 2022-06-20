import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob

import re
import string

#-----------

def single_entry_prepare(strtext):

    strsample = strtext.lower()
    strsample = re.sub("\$", "", strsample) #remove 
    strsample = re.sub("https?:\/\/.*[\r\n]*", "", strsample) #remove urls
    strsample = re.sub("#", "", strsample) #remove hashtags
    strsample = re.sub("\'\w+", '', strsample) #remove 's
    #strsample = re.sub(r'\w*\d+\w*', '', strsample) #remove numbers
    strsample = re.sub('10/10', 'SUPERLATIVE', strsample) #replace digits
    strsample = re.sub('\d', 'DIGIT', strsample) #replace digits
    strsample = re.sub('\s{2,}', " ", strsample) #remove extra spaces 
        
    return strsample

def single_entry_spell_check(strtext):
    
    textBlb = TextBlob(strtext)            # Making our first textblob
    textCorrected = textBlb.correct()   # Correcting the text
        
    return textCorrected.string

def single_entry_remove_unicode(strtext):
    
    strtext = " ".join([word for word in 
                        strtext.encode(encoding="ascii", errors="ignore").decode().split()])
           
    return strtext

def single_entry_remove_punctuation(strtext):
    
    strtext = strtext.translate(str.maketrans('', '', string.punctuation))
    
    return strtext

def single_entry_remove_stop_words(strtext, stop_words):
    
    strtext = ' '.join([word for word in strtext.split() if word not in (stop_words)])
    
    return strtext

def single_entry_lambda_stemmer(strtext):
    
    # Use English stemmer.
    #stemmer = SnowballStemmer("english")
    
    ps = PorterStemmer()
    strtext =  " ".join([ps.stem(y) for y in strtext.split()])
            
    return strtext

def single_entry_lemmatizer(strtext):
  
    lemmatizer  = WordNetLemmatizer()
    strtext = " ".join([lemmatizer.lemmatize(y) for y in strtext.split()])
            
    return strtext

#-----------

def define_stop_words(stop_words_to_add):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(stop_words_to_add)
    return stop_words


#-----------

def df_lambda_remove_stop_words(df, stop_words, column_name, new_column):
    df[new_column] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    return df

def df_lambda_to_lowercase(df, column_name, new_column):
    df[new_column] = df[column_name].apply(lambda x: x.lower())
    return df

def df_lambda_remove_punctuation(df, column_name, new_column):
    df[new_column] = df[column_name].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return df


def df_lambda_stemmer(df, column_name, new_column):
    
    # Use English stemmer.
    #stemmer = SnowballStemmer("english")
    
    ps = PorterStemmer()
    df[new_column] = df[column_name].apply(lambda x: " ".join([ps.stem(y) for y in x.split()])) 
            
    return df


def df_lambda_lemmatizer(df, column_name, new_column):
  
    lemmatizer  = WordNetLemmatizer()
    df[new_column] = df[column_name].apply(lambda x: " ".join([lemmatizer.lemmatize(y) for y in x.split()])) 
            
    return df


def df_lambda_remove_unicode(df, column_name, new_column):
    df[new_column] = df[column_name].apply(lambda x: " ".join([word for word in 
                                     x.encode(encoding="ascii", errors="ignore").decode().split()])) 
           
    return df


#-----------

def df_column_prepare(df, column_name, new_column):
    public_remarks_mentiones = []
    
    for strsample in df[column_name].values:
        strsample = strsample.lower()
        strsample = re.sub("\$", "", strsample) #remove 
        strsample = re.sub("https?:\/\/.*[\r\n]*", "", strsample) #remove urls
        strsample = re.sub("#", "", strsample) #remove hashtags
        strsample = re.sub("\'\w+", '', strsample) #remove 's
        #strsample = re.sub(r'\w*\d+\w*', '', strsample) #remove numbers
        strsample = re.sub('10/10', 'SUPERLATIVE', strsample) #replace digits
        strsample = re.sub('\d', 'DIGIT', strsample) #replace digits
        strsample = re.sub('\s{2,}', " ", strsample) #remove extra spaces 
        
        public_remarks_mentiones.append(strsample)

    df[new_column] =  public_remarks_mentiones  
        
    return df

#-----------

def df_column_simple_prepare(df, column_name, new_column):
    public_remarks_mentiones = []
    
    for strsample in df[column_name].values:
        strsample = strsample.lower()
        strsample = re.sub("\$", "", strsample) #remove 
        strsample = re.sub("https?:\/\/.*[\r\n]*", "", strsample) #remove urls
        strsample = re.sub("#", "", strsample) #remove hashtags
        strsample = re.sub("\'\w+", '', strsample) #remove 's
        strsample = re.sub('\s{2,}', " ", strsample) #remove extra spaces 
        
        public_remarks_mentiones.append(strsample)

    df[new_column] =  public_remarks_mentiones  
        
    return df


def df_spell_check(df, column_name, new_column):
    public_remarks_mentiones = []
    
    for strsample in df[column_name].values:
        textBlb = TextBlob(strsample)            # Making our first textblob
        textCorrected = textBlb.correct()   # Correcting the text
        public_remarks_mentiones.append(textCorrected)

    df[new_column] =  public_remarks_mentiones  
        
    return df


def df_column_remove_punctuation(df, column_name, new_column):
    public_remarks_punctuation = []
    
    for strsample in df[column_name].values:
        strsample = strsample.translate(str.maketrans('', '', string.punctuation))
        public_remarks_punctuation.append(strsample)

    df[new_column] =  public_remarks_punctuation  
        
    return df


def df_column_remove_unicode(df, column_name, new_column):
    public_remarks_encode = []
    
    for strtext in df[column_name].values:
        # encoding/(decoding) the text to ASCII format
        strtext = strtext.encode(encoding="ascii", errors="ignore").decode()
        clean_text = " ".join([word for word in strtext.split()])
        public_remarks_encode.append(clean_text)
        
        
    df[new_column] =  public_remarks_encode  
        
    return df


def df_column_stemmer(df, column_name, new_column):
    #SLOWER VERSION
    
    ps = PorterStemmer()
    public_remarks_stemmer = []
    
    for string in df[column_name].values:
        words = word_tokenize(string)
        for idw,word in enumerate(words): 
            words[idw] = ps.stem(word)
        string = " ".join([word for word in words])
        public_remarks_stemmer.append(string)
        
    df[new_column] =  public_remarks_stemmer  
        
    return df

