import librosa


def get_audio_array(filepath, sampling_rate=16000):
    audio_array = librosa.load(
        filepath,
        sr=sampling_rate,
        mono=False
    )[0]

    return audio_array


def get_input_for_model(audio_array, feature_extractor):
    input_values = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt"
    ).input_values

    return input_values
