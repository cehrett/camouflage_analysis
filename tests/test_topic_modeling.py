# test_topic_modeling.py
# Unit tests for topic modeling logic

import pandas as pd
from src.topic_modeling.model import preprocess_messages
from src.topic_modeling.model import topic_modeling_pipeline

def test_preprocess_messages():
    # Sample input data
    data = pd.Series([
        "@user1 Check this out! https://example.com",
        "@user2 @user3 Another message for you @user4 with a URL: http://test.com",
        "No mentions or URLs here!",
    ])

    # Expected output
    expected = pd.Series([
        "Check this out! [url]",
        "Another message for you @user4 with a URL: [url]",
        "No mentions or URLs here!",
    ])

    # Apply the preprocessing pipeline
    result = preprocess_messages(data)

    # Assert equality
    assert result.equals(expected), f"Expected {expected} but got {result}"
