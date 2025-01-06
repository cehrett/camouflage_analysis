# zero_shot.py
# Functions for zero-shot NLI classification (e.g., loading models, labeling data)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
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
    

class CamouflageClassifier:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = "cuda"):
        """
        Initialize the actionability classification pipeline.
        
        Args:
            model_name (str): Name of the pre-trained model to use.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.device = device
        self.base_messages = [
            {"role": "system", "content": "You are a helpful assistant that classifies social media posts as either \"Camouflage\" or \"Influence\".\n\n# Background:\nOnline influence campaigns make social media posts of two types. One, \"Influence\", is posts that serve the political goals of the influence campaign. These are posts that attempt to sway public opinion about some political topic of import to the organization perpetrating the influence operation. The other, \"Camouflage\", is posts that serve a secondary purpose. The goal of these posts is to make the accounts in the campaign appear more legitimate. The goal here is to make the posts appear to be coming from an authentic, genuine user, so that the same account's Influence posts will be taken more seriously by readers online. Camouflage posts might be about any topic; notably, sports.\n\n# Task:\nYour task is as follows. The user will submit a social media post. You respond with either \"Camouflage\" or \"Influence\", followed by a clear summary of why you provided that label."},
            {"role": "user", "content": "Ready for the freakin' weeken'! 🎉🍻"},
            {"role": "assistant", "content": "Camouflage. Reason: The post is a casual expression of relatable sentiment about the weekend."},
            {"role": "user", "content": "Can someone please explain to me why it's okay to call Trump racist but not Harris? ELI5"},
            {"role": "assistant", "content": "Influence. Reason: The post implies that a particular political stance is hypocritical."},
            # {"role": "user", "content": "Imagine not buying PluffCoin at this point. Why not choose to be rich? I don't get it"},
            # {"role": "assistant", "content": "Influence. Reason: The post is promoting a specific financial investment."},
            {"role": "user", "content": "Wait a sec... \"Harris, Kamala\" -> \"Harry's common, lol\" -> The Potters are not wizards -> Harry is a muggle -> The whole series is a fever dream confirmed??? makes u think 🤔"},
            {"role": "assistant", "content": "Camouflage. Reason: The post is a joking, speculative interpretation of a popular cultural reference."},
            {"role": "user", "content": "Arrrgh no, Nintendo! FPS games DO NOT BELONG IN THE ZELDA FRANCHISE! #NotMyZelda #disgusted"},
            {"role": "assistant", "content": "Camouflage. Reason: The post expresses relatable frustration about a casual matter."},
        ]

    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """
        Perform camouflage classification on a batch of texts.
        
        Args:
            texts (List[str]): List of texts to classify.
            batch_size (int): Number of texts to process in each batch.
        
        Returns:
            List[dict]: List of classification results for each text.
        """
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        with tqdm(total=total_batches, desc="Classifying batches", leave=True, bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_messages = [self.base_messages + [{"role": "user", "content": text}] for text in batch]
                
                # Tokenize the messages using chat template
                model_inputs = self.tokenizer.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    return_dict=True
                ).to(self.device)

                # Run model to get logits and generated output
                with torch.no_grad():
                    outputs = self.model.generate(
                        **model_inputs,
                        max_new_tokens=8,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

                logits = outputs.scores[0]

                # Tokenize the labels "actionable" and "not actionable" to compare logits
                labels = ["Camouflage.", "Influence."]
                label_ids = [self.tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

                # Extract logits for the target labels and apply softmax to get probabilities
                label_logits = logits[:, label_ids]
                probabilities = F.softmax(label_logits, dim=-1)

                # Get the probability for "camouflage" and "influence" for each text in the batch
                camouflage_probs = probabilities[:, 0].tolist()
                influence_probs = probabilities[:, 1].tolist()

                # Format results
                for text, cprob, iprob, generated_text in zip(batch, camouflage_probs, influence_probs, generated_texts):
                    result = {
                        "text": text,
                        "camouflage_prob": cprob,
                        "influence_prob": iprob,
                        "generated_output": generated_text.split('\n')[-1]
                    }
                    results.append(result)

                pbar.update(1)

        return pd.DataFrame(results)


def main():
    """
    Example of how to use the ZeroShotClassifier class.
    """
    # Example input data
    texts = [
        "This product will revolutionize your life!",
        "The weather today is amazing.",
        "You should vote for candidate X in the upcoming election.",
        "Grrrr can't believe they traded Pasvrainom for a 3rd round pick! Absolutely ridiculous!",
        "What if babies all had mustaches lol"
    ]
    labels = ["persuasion", "information", "opinion"]

    # Initialize classifier
    classifier = ZeroShotClassifier(model_name="facebook/bart-large-mnli", device="cuda", labels=labels)

    # Perform classification
    results = classifier.classify_batch(texts, batch_size=2)

    # Print results
    print(results)


    """
    Example of how to use the ActionabilityClassifier class.
    """

    # Initialize classifier
    classifier = CamouflageClassifier(model_name="meta-llama/Llama-3.2-3B-Instruct", device="cuda")

    # Perform classification
    results = classifier.classify_batch(texts, batch_size=2)

    # Print results
    print(results)


if __name__ == "__main__":
    main()

