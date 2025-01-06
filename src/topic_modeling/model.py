# model.py
# Logic and helper functions for unsupervised topic modeling

import pandas as pd
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP


def remove_initial_user_mentions(messages: pd.Series) -> pd.Series:
    """
    Removes all initial @[username] substrings from each message in a pandas Series.

    Args:
        messages (pd.Series): A pandas Series containing raw messages.

    Returns:
        pd.Series: A pandas Series with initial user mentions removed.
    """
    # Define a regular expression to match all initial mentions up to the first non-mention text
    pattern = r"^(?:@\w+\s*)+"

    # Apply the regex pattern to each message
    cleaned_messages = messages.apply(lambda x: re.sub(pattern, "", x.strip()))

    return cleaned_messages


def remove_urls(messages: pd.Series) -> pd.Series:
    """
    Removes all URLs from each message in a pandas Series, replacing each with '[url]' to maintain readability.

    Args:
        messages (pd.Series): A pandas Series containing raw messages.

    Returns:
        pd.Series: A pandas Series with URLs removed.
    """
    # Define a regular expression to match all URLs
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    # Apply the regex pattern to each message
    cleaned_messages = messages.apply(lambda x: re.sub(pattern, "[url]", x))

    return cleaned_messages


def preprocess_messages(messages: pd.Series) -> pd.Series:
    """
    Applies a series of preprocessing steps to clean a pandas Series of messages.

    Args:
        messages (pd.Series): A pandas Series containing raw messages.

    Returns:
        pd.Series: A pandas Series with messages cleaned and preprocessed.
    """
    # Step 1: Remove initial user mentions
    messages = remove_initial_user_mentions(messages)
    
    # Step 2: Remove URLs
    messages = remove_urls(messages)

    return messages


def perform_topic_modeling(messages: pd.Series, n_topics: int = None, min_cluster_size=10, min_samples=10, reduce_outliers=True) -> BERTopic:
    """
    Performs topic modeling on preprocessed messages using BERTopic.

    Args:
        messages (pd.Series): A pandas Series containing preprocessed messages.
        n_topics (int, optional): Number of topics to generate. Default is None (automatic selection).
        min_cluster_size (int, optional): Minimum number of messages in a cluster. Default is 10.
        min_samples (int, optional): Minimum number of samples in a cluster. Default is 10.

    Returns:
        BERTopic: A trained BERTopic model, fitted to the messages.
        pd.Series: A pandas Series of topic labels for each message.
        pd.DataFrame: A pandas DataFrame of topic probabilities for each message.
    """
    # Create instances of GPU-accelerated UMAP and HDBSCAN
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
    hdbscan_model = HDBSCAN(min_samples=min_samples, 
                            gen_min_span_tree=True, 
                            prediction_data=True,
                            min_cluster_size=min_cluster_size)

    # Initialize CountVectorizer, to remove stopwords after generating embeddings and clustering
    vectorizer_model = CountVectorizer(stop_words="english")

    # Initialize BERTopic
    topic_model = BERTopic(nr_topics=n_topics, 
                           vectorizer_model=vectorizer_model, 
                           umap_model=umap_model, 
                           hdbscan_model=hdbscan_model)

    # Fit the model to the messages
    topics, probs = topic_model.fit_transform(messages.tolist())

    # Optionally, reduce outliers
    if reduce_outliers:
        original_num_outliers = topics.count(-1)
        print("Reducing outliers. To disable this, set reduce_outliers=False.")
        topics = topic_model.reduce_outliers(messages.tolist(), topics)
        topic_model.update_topics(docs=messages.tolist(), topics=topics, vectorizer_model=vectorizer_model)
        reduced_num_outliers = topics.count(-1)
        print(f"Reduced outliers from {original_num_outliers} to {reduced_num_outliers}.")

    return topic_model, vectorizer_model


def topic_modeling_pipeline(messages: pd.Series, n_topics: int = None) -> BERTopic:
    """
    Full pipeline for topic modeling: preprocessing and topic modeling.

    Args:
        messages (pd.Series): Raw messages as a pandas Series.
        n_topics (int, optional): Number of topics to generate. Default is None.

    Returns:
        BERTopic: A trained BERTopic model.
    """
    # Preprocess messages
    preprocessed_messages = preprocess_messages(messages)

    # Perform topic modeling
    topic_model, _ = perform_topic_modeling(preprocessed_messages, n_topics=n_topics)

    return topic_model







