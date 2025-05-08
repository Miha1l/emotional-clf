from transformers import (
    HubertForSequenceClassification,
    AutoConfig,
)

from models import (
    HubertTripletClassification
)


def get_model_for_clf_train(model_dir, n_labels):
    model_id = 'facebook/hubert-base-ls960' if model_dir == '' else model_dir

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=n_labels,
    )

    is_local_file = model_dir == ''
    architecture = config.architectures[0]
    if architecture == 'HubertForTripletTrain':
        return HubertTripletClassification.from_pretrained(
            model_id,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=is_local_file,
        )

    return HubertForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=is_local_file,
    )
