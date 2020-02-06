
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:


import pandas as pd
import numpy as np

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df0 = pd.Series(doc)


# In[78]:


def date_sorter():
    
# Your code here
    df = df0
    #extract year
    df = df.str.lower()
    df = df.str.replace(',', '')
    df = df.str.replace('.', '')
    df = df.str.replace('st', '')
    df = df.str.replace('nd', '')
    df = df.str.replace('rd', '')
    df = df.str.replace('th', '')

    month_dict = {'jan': '01',
                 'feb': '02',
                  'mar': '03',
                  'apr': '04',
                  'may': '05',
                  'jun': '06',
                  'jul': '07',
                  'aug': '08',
                  'sep': '09',
                  'oct': '10',
                  'nov': '11',
                  'dec': '12'}

    month2_dict = {'january': '01',
                 'february': '02',
                  'march': '03',
                  'april': '04',
                  'may': '05',
                  'june': '06',
                  'july': '07',
                  'august': '08',
                  'september': '09',
                  'october': '10',
                  'november': '11',
                  'december': '12'}

    def clean_series(series):
         return series[series.notnull()]

    def get_next_dates(previous_dfs):
        clean_indeces = []
        for dfs in previous_dfs:
            #print(dfs.index)
            new_indeces = list(dfs.index)
            clean_indeces.extend(new_indeces)
            #print(clean_indeces)
        all_indices = df.index
        next_indeces = np.delete(all_indices, clean_indeces)
        next_dates = df[next_indeces]
        return next_dates


    def copy_index(df):
        indeces = df.index
        values = df.values
        df = pd.DataFrame()
        df['indeces'] = indeces
        df['date'] = values
        return df

    def replace_month(df):
        for month, code in month_dict.items():
            x = df['month'].str.find(month) #indexes where the substring is present
            df['month'][x>-1] = df['month'].replace(to_replace=r'\w+\W?', value=code, regex=True) #replace current month in cells with match

        df['month'] = df['month'].replace(to_replace= r'[A-Z]*[a-z]+', value=np.nan, regex=True)
        if 'day' in df.columns:
            df.loc[df['month'].isnull(), 'day'] = '01'
        df.loc[df['month'].isnull(), 'month'] = '01'
        return df

    def split_dates(series):
        dates = series.str.split('/', expand=True)
        dates.columns = ['month', 'day', 'year']
        dates['month'] = dates['month'].astype(int)
        dates['day'] = dates['day'].astype(int)
        dates['year'] = ['19'+str(i) if len(i)==2 else i for i in dates['year'] ]
        return dates

    date_dd_mm_yy = df.str.extract(r'(\d{1,2}[/-]\d{1,2}[/-]\d+)')
    date_dd_mm_yy = date_dd_mm_yy.str.replace('-', '/')
    date_dd_mm_yy = clean_series(date_dd_mm_yy)
    date_dd_mm_yy = split_dates(date_dd_mm_yy)
    next_dates = get_next_dates([date_dd_mm_yy])

    # #extract dd month year
    dates_dd_mmm_yyyy = next_dates.str.extract(r'(\d{1,2}\s[a-z]{3,10}\s[1,2]\d{3})')
    dates_dd_mmm_yyyy = clean_series(dates_dd_mmm_yyyy)
        #split 
    dates_dd_mmm_yyyy = dates_dd_mmm_yyyy.str.split(' ', expand=True)
    dates_dd_mmm_yyyy.columns = ['day', 'month', 'year']
    dates_dd_mmm_yyyy = replace_month(dates_dd_mmm_yyyy)
    dates_dd_mmm_yyyy =  dates_dd_mmm_yyyy['month'].astype(str) + '/' +dates_dd_mmm_yyyy['day'].astype(str) + '/' + dates_dd_mmm_yyyy['year'].astype(str)
    dates_dd_mmm_yyyy = split_dates(dates_dd_mmm_yyyy)
    next_dates = get_next_dates([date_dd_mm_yy, dates_dd_mmm_yyyy])

    #######

    # #extract Month date yeat
    month_day_year = next_dates.str.extract(r'([a-z]{3,10}\W?\s?\d{2}\W?\s?[1,2]\d{3})')      
    month_day_year = clean_series(month_day_year)
    month_day_year = month_day_year.str.extract(r'([a-z]{3,10})\W?\s?(\d{2})\W?\s?([1,2]\d{3})')    
    month_day_year.columns = ['month', 'day', 'year']
    month_day_year = replace_month(month_day_year)
    month_day_year = month_day_year['month'].astype(str) + '/' + month_day_year['day'].astype(str)  +  '/' + month_day_year['year'].astype(str)
    month_day_year = split_dates(month_day_year)
    next_dates = get_next_dates([date_dd_mm_yy, dates_dd_mmm_yyyy,month_day_year])



    # #extract Month  yeat
    month_year = next_dates.str.extract(r'([a-z]{3,10}\W?\s?[1,2]\d{3})')
    month_year = clean_series(month_year)
    month_year = month_year.str.split(' ', expand=True)
    month_year.columns = ['month', 'year']
    month_year = replace_month(month_year)
    month_year =  month_year['month'].astype(str) + '/01/' + month_year['year'].astype(str)
    month_year = split_dates(month_year)
    next_dates = get_next_dates([date_dd_mm_yy, dates_dd_mmm_yyyy,month_day_year,month_year])


    mm_year = next_dates.str.extract(r'([0-9]{1,2}/[1,2]\d{3})')
    mm_year = clean_series(mm_year)
    mm_year = mm_year.str.extract(r'([0-9]{1,2})/([1,2]\d{3})', expand=True)
    mm_year.columns = ['month', 'year']
    mm_year = mm_year['month'].astype(str) + '/' + '01' + '/' +  mm_year['year'].astype(str)
    mm_year = split_dates(mm_year)
    next_dates = get_next_dates([date_dd_mm_yy, dates_dd_mmm_yyyy,month_day_year,month_year,mm_year])

    yyyy = next_dates.str.extract(r'\s?\w*\W*(\d{4})\w*\W*\s?', expand=False)
    yyyy = '01/01/' + yyyy.astype(str)
    yyyy = split_dates(yyyy)

    final_df = pd.concat([date_dd_mm_yy, dates_dd_mmm_yyyy,month_day_year,month_year,mm_year, yyyy])

    length = 0
    for i in [date_dd_mm_yy, dates_dd_mmm_yyyy,month_day_year,month_year,mm_year, yyyy]:
        length += len(i)
    length

    final_df.reset_index(inplace=True)
    final_df.sort_values(by=['year', 'month', 'day'], inplace=True)
    result = final_df['index']
    result.reset_index(inplace=True, drop=True)
    return result

