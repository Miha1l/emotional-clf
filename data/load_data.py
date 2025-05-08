from datasets import (
    DatasetDict,
    load_dataset,
)

from utils import audio


def read_audio_to_array(batch, dirpath):
    batch["array"] = audio.get_audio_array(f'{dirpath}/{batch["audio"]}')
    return batch


def get_input_values(batch, feature_extractor):
    array = batch["array"]
    batch["input_values"] = audio.get_input_for_model(array, feature_extractor)[0]
    return batch


def read_triplets_to_arrays(batch, dirpath):
    batch["anchor_input_values"] = audio.get_audio_array(f'{dirpath}/{batch["anchor"]}')
    batch["positive_input_values"] = audio.get_audio_array(f'{dirpath}/{batch["positive"]}')
    batch["negative_input_values"] = audio.get_audio_array(f'{dirpath}/{batch["negative"]}')
    return batch


def get_input_values_for_triplets(batch, feature_extractor):
    keys = ["anchor_input_values", "positive_input_values", "negative_input_values"]
    for key in keys:
        array = batch[key]
        batch[key] = audio.get_input_for_model(array, feature_extractor)[0]

    return batch


def load_data_for_test(filepath, dirpath, feature_extractor):
    data_files = {"test": str(filepath)}
    ds = load_data(data_files, dirpath, feature_extractor, read_audio_to_array, get_input_values)
    ds = ds.remove_columns("array")
    return ds


def load_data_for_clf_train(filepath, dirpath, feature_extractor):
    data_files = {"train": str(filepath)}
    ds = load_data(data_files, dirpath, feature_extractor, read_audio_to_array, get_input_values)
    ds = ds.rename_column("label", "labels")
    ds = ds.remove_columns("array")

    train_val = ds["train"].train_test_split(shuffle=True, test_size=0.1)

    ds = DatasetDict({
        'train': train_val['train'],
        'val': train_val['test']
    })

    return ds


def load_data_for_triplet_train(filepath, dirpath, feature_extractor):
    data_files = {"train": str(filepath)}
    ds = load_data(data_files, dirpath, feature_extractor, read_triplets_to_arrays, get_input_values_for_triplets)

    train_val = ds["train"].train_test_split(shuffle=True, test_size=0.1)

    ds = DatasetDict({
        'train': train_val['train'],
        'val': train_val['test']
    })

    return ds


def load_data(data_files, dirpath, feature_extractor, read_audio_func, get_input_values_func):
    ds = load_dataset("csv", data_files=data_files)

    ds = ds.map(
        read_audio_func,
        fn_kwargs={"dirpath": dirpath}
    )

    ds = ds.map(
        get_input_values_func,
        fn_kwargs={"feature_extractor": feature_extractor}
    )

    return ds
