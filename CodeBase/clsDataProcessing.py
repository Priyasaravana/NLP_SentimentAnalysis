
import re
import pandas as pd
import numpy as np
import os
# Scikit-Learn packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk libraries
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
# Text processing libraries
import emoji
import re
import contractions
# Download model 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = set(stopwords.words('english')) 

class DataProcessing():
    def __init__(self):
        self.process = 'Manual'

    def get_data(self):
        # get the current directory
        current_dir = os.getcwd()
        ge_emotions = pd.read_csv(os.path.join(current_dir, 'dataset\\GE_clean.csv'))
  
        # Handling records with less than five character length
        ge_emotions = ge_emotions[~ge_emotions['Clean_text'].isin(['name', 'cad','link', 'art', 'from', 'phds','ben', 'here', 'oh', 'note','by', 'to ?', 'uh', 'at'])]

        # Handling the imbalance in the dataset
        # Since Neutral holds more than 30% of data, lets downsample them
        remove_neut = 16687 - 5000
        neut_df = ge_emotions[ge_emotions["neutral"] == 1] 
        neutral_drop_indices = np.random.choice(neut_df.index, remove_neut, replace=False)
        final_df = neut_df.drop(neutral_drop_indices)
        other_df = ge_emotions[ge_emotions["neutral"] != 1] 
        final_updated_Df = pd.concat([final_df, other_df], ignore_index=True) 
        return final_updated_Df

    def get_taxonomy(self):
        # List the emotion categorized by the team
        file_path = r"./dataset/emotions_team.txt"
        with open(file_path, "r") as file:
            google_emotions_taxonomy = file.readlines()
            google_emotions_taxonomy = [line.strip() for line in google_emotions_taxonomy]
        
        return google_emotions_taxonomy
    
    def nlp_pipeline(self, textCorpus):
        textCorpus['Clean_token'] = ''
        textCorpus['Clean_token']  = self.clean_text(textCorpus['Clean_text'])
        textCorpus['Clean_token']  = self.text_processing(textCorpus['Clean_token'])  
        textCorpus['Clean_token'] = textCorpus['Clean_token'].apply(lambda x: x.split(' '))          

        return textCorpus
    
    def clean_text(self, textCorpus):
        # Punctuations removal
        textData = textCorpus.apply(lambda x: re.sub(r"[^\w\s]","", x))
        textData = textData.apply(lambda x: re.sub(r"\s+","", x))
        textData = textData.apply(lambda x: re.sub(r"\d","", x))
        textData = textData.apply(lambda x: re.sub(r"http\S+","", x)) # Remove hyperlinks
        textData = textData.apply(lambda x: re.sub(r"[^A-Za-z0-9\s]+","", x)) # Remove special characters and puctuations
        textData = textData.apply(lambda x: re.sub(r"\b[0-9]+\b\s","", x)) # remove any stand alone numbers
        return textData

    def text_processing(self, textCorpus):        
        # Tokenize the corpus
        tokenized_emotions = textCorpus.apply(lambda desc: nltk.word_tokenize(desc))
        # initialize the WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize the word to retrieve the dictionary base words 
        # Remove stop words
        tokenized_emotions = tokenized_emotions.apply(lambda x: [lemmatizer.lemmatize(token) for token in x if token not in stop_words]) 
        tokenized_emotions = [" ".join(x) for x in tokenized_emotions]
        return tokenized_emotions

    def text_processing_exp1(self, textValue):
        # Punctuations removal
        textData = textValue.apply(lambda x: re.sub(r"[^A-Za-z_]+"," ", x))
        # Tokenize the corpus
        tokenized_emotions = textData.apply(lambda desc: nlp(desc))
        # Lemmatize the word to retrieve the dictionary base words 
        # Remove st7op words
        tokenized_emotions = tokenized_emotions.apply(lambda x: [token.lemma_ for token in x if token.lemma_ not in STOP_WORDS]) 
        tokenized_emotions = [" ".join(x) for x in tokenized_emotions] 
        return tokenized_emotions

    def countVocab_exp1(self, textCorpus):
        # Create a CountVectorizer object
        vectorizer = CountVectorizer()
        # Fit the vectorizer to the corpus
        vectorizer.fit(textCorpus)
        # Get the vocabulary (i.e., unique terms) from the vectorizer
        vocab_size = len(vectorizer.vocabulary_)
        vocab_size_model = int(0.8 * vocab_size)
        return vocab_size_model
    
    def createtf_idf_exp1(self, textCorpus, vocab_size_model):        
        # set the vocabulary of the TF-IDF to top 80% of words
        vectorizer = TfidfVectorizer(stop_words="english", max_features= vocab_size_model)
        # Fitting the vectorizer and transforming train and test data
        tfidf_emotions = vectorizer.fit_transform(textCorpus)
        vectorizer.vocabulary_.__len__()
        # Transforming from generators to arrays
        return  tfidf_emotions.toarray()
    

    def split_dataset(self, df_Emotions, ge_taxonomy, test_size_perc, val_size_perc, seq_len = 50):        
        X_train = self.pad_input(df_Emotions['Clean_TokenNo'], seq_len)
        # Converting our labels into numpy arrays
        y_train = df_Emotions.loc[:,ge_taxonomy].values
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_train, y_train, test_size=test_size_perc, random_state=42, shuffle=False)

        # Split the training set into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_perc, random_state=42, shuffle=False)

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
    def pad_input(self, sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features
    
    # Building a preprocessing function to clean text
    # https://github.com/i-benjelloun/text_emotions_detection
    
    def preprocess_corpus(self, x):
        # Adding a space between words and punctation
        x = re.sub( r'([a-zA-Z\[\]])([,;.!?])', r'\1 \2', x)
        x = re.sub( r'([,;.!?])([a-zA-Z\[\]])', r'\1 \2', x)
    
        # Demojize
        x = emoji.demojize(x)
        
        # Expand contraction
        x = contractions.fix(x)
        
        # Lower
        x = x.lower()

        #correct some acronyms/typos/abbreviations  
        x = re.sub(r"lmao", "laughing my ass off", x)  
        x = re.sub(r"amirite", "am i right", x)
        x = re.sub(r"\b(tho)\b", "though", x)
        x = re.sub(r"\b(ikr)\b", "i know right", x)
        x = re.sub(r"\b(ya|u)\b", "you", x)
        x = re.sub(r"\b(eu)\b", "europe", x)
        x = re.sub(r"\b(da)\b", "the", x)
        x = re.sub(r"\b(dat)\b", "that", x)
        x = re.sub(r"\b(dats)\b", "that is", x)
        x = re.sub(r"\b(cuz)\b", "because", x)
        x = re.sub(r"\b(fkn)\b", "fucking", x)
        x = re.sub(r"\b(tbh)\b", "to be honest", x)
        x = re.sub(r"\b(tbf)\b", "to be fair", x)
        x = re.sub(r"faux pas", "mistake", x)
        x = re.sub(r"\b(btw)\b", "by the way", x)
        x = re.sub(r"\b(bs)\b", "bullshit", x)
        x = re.sub(r"\b(kinda)\b", "kind of", x)
        x = re.sub(r"\b(bruh)\b", "bro", x)
        x = re.sub(r"\b(w/e)\b", "whatever", x)
        x = re.sub(r"\b(w/)\b", "with", x)
        x = re.sub(r"\b(w/o)\b", "without", x)
        x = re.sub(r"\b(doj)\b", "department of justice", x)
        
        #replace some words with multiple occurences of a letter, example "coooool" turns into --> cool
        x = re.sub(r"\b(j+e{2,}z+e*)\b", "jeez", x)
        x = re.sub(r"\b(co+l+)\b", "cool", x)
        x = re.sub(r"\b(g+o+a+l+)\b", "goal", x)
        x = re.sub(r"\b(s+h+i+t+)\b", "shit", x)
        x = re.sub(r"\b(o+m+g+)\b", "omg", x)
        x = re.sub(r"\b(w+t+f+)\b", "wtf", x)
        x = re.sub(r"\b(w+h+a+t+)\b", "what", x)
        x = re.sub(r"\b(y+e+y+|y+a+y+|y+e+a+h+)\b", "yeah", x)
        x = re.sub(r"\b(w+o+w+)\b", "wow", x)
        x = re.sub(r"\b(w+h+y+)\b", "why", x)
        x = re.sub(r"\b(s+o+)\b", "so", x)
        x = re.sub(r"\b(f)\b", "fuck", x)
        x = re.sub(r"\b(w+h+o+p+s+)\b", "whoops", x)
        x = re.sub(r"\b(ofc)\b", "of course", x)
        x = re.sub(r"\b(the us)\b", "usa", x)
        x = re.sub(r"\b(gf)\b", "girlfriend", x)
        x = re.sub(r"\b(hr)\b", "human ressources", x)
        x = re.sub(r"\b(mh)\b", "mental health", x)
        x = re.sub(r"\b(idk)\b", "i do not know", x)
        x = re.sub(r"\b(gotcha)\b", "i got you", x)
        x = re.sub(r"\b(y+e+p+)\b", "yes", x)
        x = re.sub(r"\b(a*ha+h[ha]*|a*ha +h[ha]*)\b", "haha", x)
        x = re.sub(r"\b(o?l+o+l+[ol]*)\b", "lol", x)
        x = re.sub(r"\b(o*ho+h[ho]*|o*ho +h[ho]*)\b", "ohoh", x)
        x = re.sub(r"\b(o+h+)\b", "oh", x)
        x = re.sub(r"\b(a+h+)\b", "ah", x)
        x = re.sub(r"\b(u+h+)\b", "uh", x)

        # Handling emojis
        x = re.sub(r"<3", " love ", x)
        x = re.sub(r"xd", " smiling_face_with_open_mouth_and_tightly_closed_eyes ", x)
        x = re.sub(r":\)", " smiling_face ", x)
        x = re.sub(r"^_^", " smiling_face ", x)
        x = re.sub(r"\*_\*", " star_struck ", x)
        x = re.sub(r":\(", " frowning_face ", x)
        x = re.sub(r":\^\(", " frowning_face ", x)
        x = re.sub(r";\(", " frowning_face ", x)
        x = re.sub(r":\/",  " confused_face", x)
        x = re.sub(r";\)",  " wink", x)
        x = re.sub(r">__<",  " unamused ", x)
        x = re.sub(r"\b([xo]+x*)\b", " xoxo ", x)
        x = re.sub(r"\b(n+a+h+)\b", "no", x)

        # Handling special cases of text
        x = re.sub(r"h a m b e r d e r s", "hamberders", x)
        x = re.sub(r"b e n", "ben", x)
        x = re.sub(r"s a t i r e", "satire", x)
        x = re.sub(r"y i k e s", "yikes", x)
        x = re.sub(r"s p o i l e r", "spoiler", x)
        x = re.sub(r"thankyou", "thank you", x)
        x = re.sub(r"a^r^o^o^o^o^o^o^o^n^d", "around", x)

        # Remove special characters and numbers replace by space + remove double space
        x = re.sub(r"\b([.]{3,})"," dots ", x)
        x = re.sub(r"[^A-Za-z!?_]+"," ", x)
        x = re.sub(r"\b([s])\b *","", x)
        x = re.sub(r" +"," ", x)
        x = x.strip()
        return x