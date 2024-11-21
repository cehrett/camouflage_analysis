# zero_shot.py
# Functions for zero-shot classification (e.g., loading models, labeling data)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm  
from typing import List

class ZeroShotClassifier:
    """
    A class for performing zero-shot classification on text data.
    """

    def __init__(self, 
                 model_name: str = "facebook/bart-large-mnli", 
                 device: str = "cuda",
                 labels: List[str] = None):
        """
        Initialize the zero-shot classification pipeline.
        
        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): Device to run the model on ('cpu' or 'cuda').
            labels (List[str]): List of candidate labels for classification.
        
        Raises:
            ValueError: If `labels` is not provided.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device
        self.labels = labels
        if self.labels is None:
            raise ValueError("Labels must be provided.")
        self.results = None

    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        """
        Perform zero-shot classification on a batch of texts using proper NLI input formatting.
        
        Args:
            texts (List[str]): List of texts to classify.
            batch_size (int): Number of texts to process in each batch.
        
        Returns:
            List[dict]: List of classification results for each text.
        """
        # Dynamically determine the index for "entailment" from the model configuration
        entailment_index = {label: idx for idx, label in self.model.config.id2label.items()}["entailment"]
        
        results = []

        # Initialize tqdm progress bar
        total_batches = (len(texts) + batch_size - 1) // batch_size  # Total number of batches
        with tqdm(total=total_batches, desc="Classifying batches", leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Prepare premise-hypothesis pairs for all text-label combinations
                premise_hypothesis_pairs = [
                    (text, f"This tweet is an example of {label}.")
                    for text in batch
                    for label in self.labels
                ]
                
                # Pre-tokenize the batch
                inputs = self.tokenizer(
                    [pair[0] for pair in premise_hypothesis_pairs],  # Premises
                    [pair[1] for pair in premise_hypothesis_pairs],  # Hypotheses
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Perform a forward pass through the model
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Extract entailment scores using the dynamic index
                logits = outputs.logits[:, entailment_index].view(len(batch), len(self.labels)).softmax(dim=-1)
                
                # Format results
                for text, scores in zip(batch, logits):
                    result = {
                        "text": text,
                        "labels": self.labels,
                        "scores": scores.tolist(),
                    }
                    results.append(result)

                # Update progress bar
                pbar.update(1)
        
        self.results = results
        self.results_df = self._results_to_dataframe(results)
        return self.results_df

    def _results_to_dataframe(self, results):
        """
        Convert classification results into a pandas DataFrame.
        
        Args:
            results (list): A list of dictionaries, where each dictionary contains:
                            - 'text': The input text.
                            - 'labels': A list of label strings.
                            - 'scores': A list of corresponding scores for each label.
        
        Returns:
            pd.DataFrame: A DataFrame with columns for 'text' and one column for each label.
        """
        # Initialize lists to store data for the DataFrame
        texts = []
        label_scores = {label: [] for label in results[0]['labels']}  # Create columns for each label
        
        # Process each result
        for result in results:
            texts.append(result['text'])  # Add the text
            
            # Add scores for each label
            for label, score in zip(result['labels'], result['scores']):
                label_scores[label].append(score)
        
        # Combine into a DataFrame
        df = pd.DataFrame({'text': texts})
        for label, scores in label_scores.items():
            df[label] = scores
        
        return df




def main():
    """
    Example of how to use the ZeroShotClassifier class.
    """
    # Example input data
    texts = [
        "This product will revolutionize your life!",
        "The weather today is amazing.",
        "You should vote for candidate X in the upcoming election.",
    ]
    labels = ["persuasion", "information", "opinion"]

    # Initialize classifier
    classifier = ZeroShotClassifier(model_name="facebook/bart-large-mnli", device="cuda", labels=labels)

    # Perform classification
    results = classifier.classify_batch(texts, batch_size=2)

    # Print results
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Labels: {result['labels']}")
        print(f"Scores: {result['scores']}\n")


if __name__ == "__main__":
    main()

