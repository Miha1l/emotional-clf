from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2Processor

import torch


def batch_padding(input_values: List[Dict[str, List[float]]], max_len: int) -> torch.Tensor:
    padded_input_values = []
    for in_values in input_values:
        padded_input = in_values["input_values"].copy()

        arr_len = len(padded_input)

        for _ in range(1, max_len // arr_len):
            padded_input.extend(in_values["input_values"])

        padded_input.extend(in_values["input_values"][:(max_len % arr_len)])
        padded_input_values.append(padded_input)

    return torch.tensor(padded_input_values)


def get_len(x):
    return len(x["input_values"])


@dataclass
class DataCollatorForClassification:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: bool = False
    max_length: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        max_len = max(map(get_len, input_features))

        batch = batch_padding(input_features, max_len)

        return {
            "input_values": batch,
            "labels": torch.tensor(labels),
        }


@dataclass
class DataCollatorForTriplets:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: bool = False
    max_length: Optional[int] = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        anchor_input = [{"input_values": feature["anchor_input_values"]} for feature in features]
        positive_input = [{"input_values": feature["positive_input_values"]} for feature in features]
        negative_input = [{"input_values": feature["negative_input_values"]} for feature in features]

        max_anchor_len = max(map(get_len, anchor_input))
        max_positive_len = max(map(get_len, positive_input))
        max_negative_len = max(map(get_len, negative_input))

        max_len = max(max(max_anchor_len, max_positive_len), max_negative_len)

        anchor = batch_padding(anchor_input, max_len)
        positive = batch_padding(positive_input, max_len)
        negative = batch_padding(negative_input, max_len)

        return {
            'anchor_input_values': anchor,
            'positive_input_values': positive,
            'negative_input_values': negative,
        }
