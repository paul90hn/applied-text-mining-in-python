
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[1]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('book')
from nltk.book import *

import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# In[9]:





# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[12]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[14]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()

len(set(moby_tokens))


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[21]:


from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[27]:


def answer_one():
    total_words = len(nltk.word_tokenize(moby_raw))
    unique_words = len(set(nltk.word_tokenize(moby_raw)))
    ratio = unique_words / total_words
    
    return ratio # Your answer here

answer_one()
#correcta


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[10]:


def answer_two():
    tokens = nltk.word_tokenize(moby_raw)
    whales = [w for w in tokens if (w== 'whale' or w=='Whale')]
    percentage = (len(whales)/len(tokens))*100
    return percentage # Your answer here  0.42593747426137496

answer_two()

#correcta


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[14]:


def answer_three():
    tokens = nltk.word_tokenize(moby_raw)
    frequency = FreqDist(tokens)

    words = frequency.keys()
    values = list(frequency.values())

    frequency = pd.Series(values, index=words)
    frequency.sort_values(ascending=False, inplace=True)
    top_tokens = frequency.head(20)
    top_values = top_tokens.values
    top_tokens = top_tokens.index
    
    return list(zip(top_tokens, top_values)) # Your answer here 

answer_three()
#correcta


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[89]:


def answer_four():
    tokens = nltk.word_tokenize(moby_raw)
    frequency = FreqDist(tokens)

    long_words = [word for word in frequency if (frequency[word]>150 & len(word)>5)]
    long_words = sorted(long_words)
    frequency = pd.DataFrame.from_dict(frequency)
    return # Your answer here

#answer_four()

tokens = nltk.word_tokenize(moby_raw)
frequency = FreqDist(tokens)

long_words = [word for word in frequency if (frequency[word]>150 & len(word)>5)]
long_words = sorted(long_words)
frequency = pd.DataFrame.from_dict(frequency, orient='index')
frequency.reset_index(inplace=True)

frequency.columns = ['token', 'frequency']
frequency = frequency[frequency['frequency']> 150]
length = []
for token in frequency['token']:
    length.append(len(token))
    
frequency['length'] = length
frequency = frequency[frequency['length']>5]
frequency.sort_values(by=['token'], inplace=True)
sorted(frequency['token'])


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[54]:


def answer_five():
    tokens = nltk.word_tokenize(moby_raw.lower())
    lengths = [len(word) for word in tokens]
    df = pd.DataFrame()
    df['token'] = tokens
    df['length'] = lengths
    lengths = df
    max_index = lengths['length'].idxmax()
    max_length = lengths['length'][max_index]
    max_word = lengths['token'][max_index]
    
    
    return (max_word, max_length) # Your answer here

answer_five()

#correcta


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[20]:


def answer_six():
    tokens = nltk.word_tokenize(moby_raw)
    frequency = FreqDist(tokens)
    alphanumeric = [word for word in frequency.keys() if (word.isalpha()==True)]
    plus_2000 = [word for word in alphanumeric if frequency[word]>2000]
    frequencies_plus_2000 = [frequency[word] for word in plus_2000]
    frequencies = pd.DataFrame()
    frequencies['word'] = plus_2000
    frequencies['frequency'] = frequencies_plus_2000
    frequencies.sort_values(by=['frequency'], ascending=False, inplace=True)
    top_words = frequencies['word']
    top_frequencies = frequencies['frequency']

    return list(zip(top_frequencies, top_words))# Your answer here

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[44]:


def answer_seven():
    tokens = nltk.word_tokenize(moby_raw.lower())
    sentences = nltk.sent_tokenize(moby_raw.lower())
    tokens_per_sentence = len(tokens)/ len(sentences)
    
    return tokens_per_sentence # Your answer here

answer_seven()

#correcta


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[22]:


def answer_eight():
    tokens = nltk.word_tokenize(moby_raw)
    tags = nltk.pos_tag(tokens)

    parts = [word[1] for word in tags]
    frequency = FreqDist(parts)
    frequency = pd.DataFrame.from_dict(frequency, orient='index')
    frequency.columns = ['frequency']
    frequency.sort_values(by=['frequency'], ascending=False, inplace=True)
    frequency = frequency.head(5)
    part_of_speech = frequency.index
    frequency = frequency['frequency'].values
    
    result = []
    for i in range(len(part_of_speech)):
        result.append((part_of_speech[i], frequency[i]))  
    
    return list(zip(part_of_speech, frequency)) # Your answer here  [('NN', 39860), ('IN', 28831), ('DT', 26033), ('JJ', 19562), (',', 19204)]

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[2]:


from nltk.corpus import words

correct_spellings = words.words()
len(correct_spellings)


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[73]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    def get_recomendation(entry, n):
    
        distances = {}
        entry_gram = set(nltk.ngrams(entry, n=n))
        first_e = entry[0]
        for word in correct_spellings:
            first_w = word[0]

            if first_e == first_w :
                word_gram = set(nltk.ngrams(word, n=n))        
                jaccard_distance = nltk.jaccard_distance(entry_gram, word_gram)
                distances[word] = jaccard_distance

        distances = pd.DataFrame.from_dict(distances, orient='index')
        distances.columns = ['distance']
        distances.sort_values(by= ['distance'], ascending=True, inplace=True) #, axis=1, inplace=True)
        result = distances.index[0]
        return result

    results = []
    for entry in entries:
        results.append(get_recomendation(entry, 3))

    return results# Your answer here
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[75]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    def get_recomendation(entry, n):
    
        distances = {}
        entry_gram = set(nltk.ngrams(entry, n=n))
        first_e = entry[0]
        for word in correct_spellings:
            first_w = word[0]

            if first_e == first_w :
                word_gram = set(nltk.ngrams(word, n=n))        
                jaccard_distance = nltk.jaccard_distance(entry_gram, word_gram)
                distances[word] = jaccard_distance

        distances = pd.DataFrame.from_dict(distances, orient='index')
        distances.columns = ['distance']
        distances.sort_values(by= ['distance'], ascending=True, inplace=True) #, axis=1, inplace=True)
        result = distances.index[0]
        return result

    results = []
    for entry in entries:
        results.append(get_recomendation(entry, 4))
        
    return results# Your answer here
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[76]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    def get_recomendation(entry, n):
    
        distances = {}
        #entry_gram = set(nltk.ngrams(entry, n=n))
        first_e = entry[0]
        for word in correct_spellings:
            first_w = word[0]

            if first_e == first_w :
                #word_gram = set(nltk.ngrams(word, n=n))        
                edit_distance = nltk.edit_distance(entry, word)
                distances[word] = edit_distance

        distances = pd.DataFrame.from_dict(distances, orient='index')
        distances.columns = ['distance']
        distances.sort_values(by= ['distance'], ascending=True, inplace=True) #, axis=1, inplace=True)
        result = distances.index[0]
        return result

    results = []
    for entry in entries:
        results.append(get_recomendation(entry, 4))
        
    
    
    return results # Your answer here 
    
answer_eleven()


# In[ ]:




