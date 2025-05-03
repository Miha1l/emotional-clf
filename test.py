from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)

from models import (
    compute_metrics,
)

from data import (
    load_data_for_test,
)

from tqdm import tqdm

import numpy as np
import torch
import json


def test_model(filepath, dirpath, model_dir):
    model_id = "facebook/hubert-base-ls960"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    model = HubertForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    data = load_data_for_test(filepath, dirpath, feature_extractor)
    labels = np.array(data['test']['label'])
    logits = []

    with torch.no_grad():
        for row in tqdm(data['test'], ncols=100):
            logits.append(model(torch.tensor([row['input_values']])).logits)

    metrics = compute_metrics((logits, labels))
    print(metrics)

    metrics['model'] = model_dir
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print('Результаты сохранены в файл metrics.json')
