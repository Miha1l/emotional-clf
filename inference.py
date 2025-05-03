from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor
)

from utils import (
    get_input_for_model,
    get_audio_array
)

from tqdm import tqdm

import torch
import os
import glob


BIN_LABELS = ["positive", "sad"]
FULL_LABELS = ["angry", "neutral", "positive", "sad"]


def label_to_emotion(label, n_classes=2):
    if n_classes == 2:
        return BIN_LABELS[label]

    return FULL_LABELS[label]


def predict(logits):
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = label_to_emotion(predictions.numpy()[0])
    return predicted_emotion


def make_predicts(dirpath, model_path):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )

    print('Начало обработки')

    predicts = []
    abs_path = os.path.abspath(dirpath)
    for filepath in tqdm(glob.glob(f'{abs_path}/*.wav'), ncols=100):
        audio_array = get_audio_array(filepath)
        input_values = get_input_for_model(audio_array, feature_extractor)

        with torch.no_grad():
            logits = model(input_values).logits

        emotion = predict(logits)
        predicts.append({'file': os.path.basename(filepath), 'emotion': emotion})

    return predicts
