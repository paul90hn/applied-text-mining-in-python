
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[187]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[188]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[189]:


def answer_one():
    is_spam = np.sum(spam_data['target']==1)
    total = spam_data.shape[0]
    result = is_spam/total
    result = result*100
    
    return result #Your answer here



# In[190]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[191]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    model = CountVectorizer().fit(X_train)
    tokens = model.get_feature_names()
    max_lenght = max([len(token) for token in tokens])
    max_token = [token for token in tokens if len(token)== max_lenght][0]
    return max_token#Your answer here


# In[192]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[193]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    count_vectorizer = CountVectorizer().fit(X_train)
    transformed_train = count_vectorizer.transform(X_train)
    transformed_test = count_vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1).fit(transformed_train, y_train)
    prediction = model.predict(transformed_test)
    result = roc_auc_score(y_test, prediction)
    return result #Your answer here


# In[194]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[195]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vectorizer = TfidfVectorizer().fit(X_train)
    transformed_data = vectorizer.transform(X_train)
    feature_names = np.array(vectorizer.get_feature_names())

    max_tf_idfs = transformed_data.max(0).toarray()[0]
    sorted_indices = max_tf_idfs.argsort() #sort indices from smallest to largest
    
    bot_20_features = feature_names[sorted_indices[:20]]
    bot_20_scores = max_tf_idfs[sorted_indices[:20]]
    bot_20 = pd.Series(bot_20_scores, index=bot_20_features)

    top_20_features = feature_names[sorted_indices[:-21: -1]]
    top_20_values = max_tf_idfs[sorted_indices[:-21: -1]]
    top_20 = pd.Series(top_20_values, index=top_20_features)


    return (bot_20, top_20) #Your answer here


# In[196]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[197]:


def answer_five():
    vectorizer = TfidfVectorizer(min_df=3).fit(X_train)
    transformed_train = vectorizer.transform(X_train)
    transformed_test = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1).fit(transformed_train, y_train)
    prediction = model.predict(transformed_test)
    result = roc_auc_score(y_test, prediction)
    
    return result #Your answer here


# In[198]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[199]:


def answer_six():
    spam = spam_data[spam_data['target']==1]
    not_spam = spam_data[spam_data['target']==0]

    spam_lenght = [len(text) for text in spam['text']]
    total_spam_lenght = np.mean(spam_lenght)
    not_spam_lenght = [len(text) for text in not_spam['text']]
    total_not_spam_lenght = np.mean(not_spam_lenght)
    return (total_not_spam_lenght, total_spam_lenght) #Your answer here


# In[200]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[201]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[202]:


from sklearn.svm import SVC

def answer_seven():
    vectorizer = TfidfVectorizer(min_df=5).fit(X_train)
    transformed_train = vectorizer.transform(X_train)
    transformed_test = vectorizer.transform(X_test)

    length_train = X_train.str.len() #[len(text) for text in X_train]
    length_test = X_test.str.len()

    transformed_train = add_feature(transformed_train, length_train)
    transformed_test= add_feature(transformed_test, length_test)

    model = SVC(C=10000).fit(transformed_train, y_train)
    prediction = model.predict(transformed_test)
    result = roc_auc_score(y_test, prediction)

    return result#Your answer here


# In[203]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[204]:



def answer_eight():
    import re
    spam = spam_data[spam_data['target']==1]
    not_spam = spam_data[spam_data['target']==0]

    spam['count'] = [len(re.findall(r'\d', string)) for string in spam['text']]
    not_spam['count'] = [len(re.findall(r'\d', string)) for string in not_spam['text']]

    spam_average = np.mean(spam['count'])
    not_spam_average = np.mean(not_spam['count'])
    
    return (not_spam_average, spam_average)#Your answer here


# In[205]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[206]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    import re
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    transformed_train = vectorizer.transform(X_train)
    transformed_test = vectorizer.transform(X_test)


    X_train_n_digist = [len(re.findall(r'\d', string)) for string in X_train] #digists per document
    X_test_n_digist = [len(re.findall(r'\d', string)) for string in X_test]

    train_len = [len(text) for text in X_train]
    test_len = [len(text) for text in X_test]

    transformed_train = add_feature(transformed_train, X_train_n_digist)
    transformed_train = add_feature(transformed_train, train_len)

    transformed_test = add_feature(transformed_test, X_test_n_digist)
    transformed_test = add_feature(transformed_test, test_len)

    model = LogisticRegression(C=100).fit(transformed_train, y_train)
    prediction = model.predict(transformed_test)
    result = roc_auc_score(y_test, prediction)
    return float(result) #Your answer here


# In[207]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[208]:


def answer_ten():
    import re
    spam = spam_data[spam_data['target']==1]
    not_spam = spam_data[spam_data['target']==0]

    spam['count'] = [len(re.findall('\W', string)) for string in spam['text']]
    not_spam['count'] = [len(re.findall('\W',string)) for string in not_spam['text']]

    spam_average = np.mean(spam['count'])
    not_spam_average = np.mean(not_spam['count'])
    
    return (not_spam_average, spam_average)#Your answer here


# In[209]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[263]:


def answer_eleven():
    import re
    count_vectorizer = CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=(2,5)).fit(X_train)
    transformed_train = count_vectorizer.transform(X_train)
    transformed_test = count_vectorizer.transform(X_test)

    train_len = [len(text) for text in X_train]
    test_len = [len(text) for text in X_test]

    train_digits = [len(re.findall(r'\d', string)) for string in X_train]
    test_digits = [len(re.findall(r'\d', string)) for string in X_test]

    train_no_words = [len(re.findall('\W', string)) for string in X_train]
    test_no_words = [len(re.findall('\W', string)) for string in X_test]

    transformed_train = add_feature(transformed_train, train_len)
    transformed_train = add_feature(transformed_train, train_digits)
    transformed_train = add_feature(transformed_train, train_no_words)

    transformed_test = add_feature(transformed_test, test_len)
    transformed_test = add_feature(transformed_test, test_digits)
    transformed_test = add_feature(transformed_test, test_no_words)

    model = LogisticRegression(C=100).fit(transformed_train, y_train)
    prediction = model.predict(transformed_test)
    auc_score = roc_auc_score(y_test, prediction)


    features = np.array(count_vectorizer.get_feature_names()) 
    features = np.append(features, ['length_of_doc', 'digit_count', 'non_word_char_count'])
    coef = model.coef_[0]
    sorted_indices = coef.argsort() #sort indices from smallest to largest

#     bot_10_features = features[sorted_indices[:10]]
#     bot_10_coef = coef[sorted_indices[:10]]
#     bot_10 = pd.Series(bot_10_coef, index=bot_10_features)

#     top_10_features = features[indices_order[:-11: -1]]
#     top_10_coef = coef[sorted_indices[:-11: -1]]
#     top_10 = pd.Series(top_10_coef, index=top_10_features)

    df = pd.DataFrame()
    df['features'] = features
    df['coef'] = coef
    df.sort_values(by=['coef'], axis=0, ascending=True, inplace=True)

    bot_df =df.head(10)
    index = bot_df['features'].values
    values = bot_df['coef'].values
    bot_10=  pd.Series(values, index=index)

    top_df = df.tail(10)
    top_df.sort_values(by=['coef'], axis=0, inplace=True, ascending=False)
    index = top_df['features'].values
    values = top_df['coef'].values
    top_10 = pd.Series(values, index=index)


    return (auc_score, bot_10, top_10) #Your answer here



# In[264]:


answer_eleven()


# In[ ]:




