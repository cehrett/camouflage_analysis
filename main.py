# main.py
# Central script to run the full analysis pipeline

import pandas as pd
from bertopic import BERTopic
from visualization.plots import generate_html_report
from topic_modeling.model import topic_modeling_pipeline

def main(data_file, message_column="Message", output_path=None, inline=False, n_topics=None, nr_bins=80, label_file=None, dev_test=False):
    # Load the raw data
    raw_data = pd.read_csv(data_file)
    
    # Subset to the first 8000 rows if dev_test is True
    if dev_test:
        raw_data = raw_data.head(8000)
    
    # Ensure the data has the specified message column
    if message_column not in raw_data.columns:
        raise ValueError(f"The input data file must contain a '{message_column}' column.")
    
    # Extract the messages
    messages = raw_data[message_column]
    
    # Process the label file if provided
    labels = None
    if label_file:
        labels = generate_labels_from_csv(label_file, dev_test)
    
    # Perform topic modeling
    topic_model = topic_modeling_pipeline(messages, n_topics=n_topics)
    
    # Generate the HTML report
    report_path = generate_html_report(topic_model, output_path=output_path, classes=labels, inline=inline, docs=messages, timestamps=None, nr_bins=nr_bins)
    
    if report_path:
        print(f"Report saved to: {report_path}")
    else:
        print("Report displayed inline.")

def generate_labels_from_csv(label_file, dev_test=False):
    """
    Generates labels for each row in the CSV file based on the largest numeric element in each row.

    Args:
        label_file (str): Path to the CSV file.
        dev_test (bool): Whether to subset the CSV to the first 1000 rows.

    Returns:
        list: A list of labels for each row.
    """
    df = pd.read_csv(label_file)
    
    # Subset to the first 1000 rows if dev_test is True
    if dev_test:
        df = df.head(8000)
    
    # Identify numeric columns once
    numeric_cols = df.select_dtypes(include='number').columns
    
    labels = []
    for _, row in df.iterrows():
        numeric_values = row[numeric_cols]
        if not numeric_values.empty:
            max_col = numeric_values.idxmax()
            labels.append(max_col)
        else:
            labels.append(None)
    
    return labels

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a topic modeling report from a dataset.")
    parser.add_argument("data_file", type=str, help="Path to the raw data file (CSV format).")
    parser.add_argument("--message_column", type=str, default="Message", help="Name of the column containing the messages. Default is 'Message'.")
    parser.add_argument("--output_path", type=str, default=None, help="Directory to save the HTML report. Defaults to the current working directory.")
    parser.add_argument("--inline", action="store_true", help="Display the report inline in a Jupyter notebook.")
    parser.add_argument("--n_topics", type=int, default=None, help="Number of topics to generate. Default is None (automatic selection).")
    parser.add_argument("--nr_bins", type=int, default=80, help="Number of bins for the 'Topics Over Time' visualization. Default is 80.")
    parser.add_argument("--label_file", type=str, default=None, help="Path to the second CSV file for generating labels.")
    parser.add_argument("--dev_test", action="store_true", help="Subset the input CSV(s) to just the first 1000 rows.")
    
    args = parser.parse_args()
    
    main(args.data_file, 
         message_column=args.message_column, 
         output_path=args.output_path, 
         inline=args.inline, 
         n_topics=args.n_topics, 
         nr_bins=args.nr_bins,
         label_file=args.label_file,
         dev_test=args.dev_test)