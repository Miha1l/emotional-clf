from transformers import (
    Wav2Vec2FeatureExtractor,
)

from models import (
    compute_metrics,
    metrics_plot,
    get_model_for_test,
)

from data import (
    load_data_for_test,
)

from tqdm import tqdm

import numpy as np
import torch
import json


def test_model(filepath, dirpath, model_dir, output):
    model_id = "facebook/hubert-base-ls960"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = get_model_for_test(model_dir)

    data = load_data_for_test(filepath, dirpath, feature_extractor)
    labels = np.array(data['test']['label'])
    logits = []

    with torch.no_grad():
        for row in tqdm(data['test'], ncols=100):
            logits.append(model(torch.tensor([row['input_values']])).logits)

    metrics = compute_metrics((logits, labels))
    print(metrics)

    metrics['model'] = model_dir
    with open(output, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Результаты сохранены в файл {output}')

    predictions = np.argmax(logits, axis=-1)
    metrics_plot(labels, predictions)
