import pandas as pd
import os

os.chdir('/home/cehrett/Projects/Trolls/baseball_tweets')

df = pd.read_csv(os.path.join('camouflage_analysis','data','mlb_full_hist.csv'), index_col=0)

import camouflage_analysis.src.topic_modeling.model as topic_modeling

cm = topic_modeling.preprocess_messages(df['Message'])

import camouflage_analysis.src.classification.zero_shot as zero_shot

LLM = zero_shot.CamouflageClassifier()

results = LLM.classify_batch(cm.tolist(), batch_size=32)

results.to_csv(os.path.join('camouflage_analysis','data','camouflage_classification_results.csv'), index=False)