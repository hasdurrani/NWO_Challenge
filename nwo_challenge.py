#!/usr/bin/env python3

# Importing packages
import sys
import re
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from ordered_set import OrderedSet

def make_query_reddit(query_term, limit):
    """Function to query the reddit dataset for a particular 
    term and return the data as a pandas dataframe"""
    
    # Authentication
    client = bigquery.Client.from_service_account_json('./nwo-sample-5f8915fdc5ec.json')

    #Perform a query.
    QUERY = (
        'SELECT * FROM `nwo-sample.graph.reddit` '
        'WHERE LOWER(body) LIKE "%{}%" '
        'ORDER BY created_utc DESC '
        'LIMIT {}').format(query_term.lower(), limit)

    query_job = client.query(QUERY)  # API request

    return query_job.to_dataframe()

def make_query_twitter(query_term, limit):
    """Function to query the twitter dataset for a particular 
    term and return the data as a pandas dataframe"""
    
    # Authentication
    client = bigquery.Client.from_service_account_json('./nwo-sample-5f8915fdc5ec.json')

    #Perform a query.
    QUERY = (
        'SELECT * FROM `nwo-sample.graph.tweets` '
        'WHERE LOWER(tweet) LIKE "%{}%" '
        'ORDER BY created_at DESC '
        'LIMIT {}').format(query_term.lower(), limit)

    query_job = client.query(QUERY)  # API request

    return query_job.to_dataframe()

def clean_data_reddit(df_reddit):
    """Function to clean the reddit dataset by removing URLs, 
    instances of ".com", and combining the "subreddit" and "body" 
    columns into a "text"
    The function returns a smaller dataset with only the text column"""
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    df_reddit['body'] = df_reddit.body.apply(lambda x: url_pattern.sub(r'', x))

    df_reddit['body'] = df_reddit.body.apply(lambda x: x.replace('.com', ''))

    df_reddit['text'] = df_reddit.subreddit.fillna('') + ' ' + df_reddit.body.fillna('')

    df_reddit = df_reddit[['text']]
    
    return df_reddit
    
def clean_data_twitter(df_twitter):
    """Function to clean the twitter dataset by removing URLs, 
    instances of ".com", and combining the "hashtags" and "tweet" 
    columns into a "text"
    The function returns a smaller dataset with only the text column"""

    df_twitter['hashtags'] = df_twitter.hashtags.apply(lambda x: ' '.join(x))

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    df_twitter['tweet'] = df_twitter.tweet.apply(lambda x: url_pattern.sub(r'', x))

    df_twitter['tweet'] = df_twitter.tweet.apply(lambda x: x.replace('.com', ''))

    df_twitter['text'] = df_twitter.hashtags.fillna('') + ' ' + df_twitter.tweet.fillna('')

    df_twitter = df_twitter[['text']]
    
    return df_twitter
    
def generate_co_occurence(df_text_column):    
    """This function takes the text column of a dataset as an input and generates a co-occurence matrix, 
    which is then returned as a pandas dataframe to preserve feature names"""
    
    count_vectorizer = CountVectorizer(stop_words = 'english', ngram_range=(1, 2), lowercase=True)
    vectorized_matrix = count_vectorizer.fit_transform(df_text_column)

    co_occurrence_matrix = (vectorized_matrix.T * vectorized_matrix)
    co_occurrence_matrix.setdiag(0)
    
    return pd.DataFrame(co_occurrence_matrix.A, 
                        columns=count_vectorizer.get_feature_names(),
                        index=count_vectorizer.get_feature_names())

 
def generate_graph(df_com):
    """This function takes a pandas dataframe representation of a co-occurence matrix, 
    and converts it into a graph where the word co-occurence is represented as nodes and edges"""

    df_com.values[np.tril(np.ones(df_com.shape)).astype(np.bool)] = 0
    
    df_com_stacked = df_com.stack()

    df_com_stacked = df_com_stacked[df_com_stacked >= 1].rename_axis(('source', 'target')).reset_index(name='weight')
    
    graph = nx.from_pandas_edgelist(df_com_stacked,  edge_attr=True)
    
    return graph

def find_neighbors(graph, query_term, topn):
    """This function takes a graph and a query term as input, and finds the nodes 
    associated with the query term of interest, and returns an ordered list of 
    it's neighboring nodes (trends). 
    The list is sorted based on the edge weight (descending) which comes from the Co-Occurence Matrix.
    A second parameter for sorting is the uniqueness of the association based on how many 
    other nodes the neighbor is connected to (descending)"""
    
    node_of_interest = query_term.lower()
    
    neighbors = []

    for neighbor in graph.neighbors(node_of_interest):
        neighbors.append((neighbor, graph.get_edge_data(node_of_interest, neighbor)['weight'], len(graph.edges(neighbor))))

    neighbors.sort(key=lambda x:(-x[1], x[2]))

    return OrderedSet([neighbor[0] for neighbor in neighbors[:topn]])

# Gather our code in a main() function
def main():
    
    QUERY_TERM = sys.argv[1]
  
    # Loading the data
    df_reddit = make_query_reddit(QUERY_TERM, 500)
    df_twitter = make_query_twitter(QUERY_TERM, 500)
    print('data loaded')

    # Cleaning the individual datasets
    df_reddit = clean_data_reddit(df_reddit)
    df_twitter = clean_data_twitter(df_twitter)
    print('data cleaned')

    # Merging the individual datasets into a single dataset
    df = pd.concat([df_reddit, df_twitter], ignore_index=True)
    print('datasets merged')

    # Generating a Co-Occurence Matrix
    df_com = generate_co_occurence(df['text'])
    print('co-occurence matrix generated')

    # Generating a Graph representation of the Co-Occurence Matrix
    print('generating the graph - please be patient')
    G = generate_graph(df_com)
    print('graph generated')

    # Outputting topn (5) most closely associated trends
    print("Top 5 closely associated keywords:\n{}".format(find_neighbors(G, QUERY_TERM, 6)))

if __name__ == '__main__':
    main()