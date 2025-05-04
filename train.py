from transformers import (
    HubertForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)

from models import (
    HubertForTripletTrain,
    HubertClassificationAfterTriplet,
    compute_metrics,
)

from data import (
    DataCollatorForClassification,
    DataCollatorForTripletTrain,
    load_data_for_clf_train,
    load_data_for_triplet_train,
)

import torch.cuda


def unfreeze_model_layers(model, n_layers):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in list(model.named_parameters())[-n_layers:]:
        param.requires_grad = True


def triplet_train(filepath, dirpath, output_dir, n_epochs, device):
    model_id = "facebook/hubert-base-ls960"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(
        model_id,
    )

    model = HubertForTripletTrain.from_pretrained(
        model_id,
        config=config,
    )

    unfreeze_model_layers(model, 2 + config.num_hidden_layers * 16)

    if not torch.cuda.is_available() and device == 'gpu':
        print('CUDA недоступна, обучение будет происходить на CPU')
        device = 'cpu'

    training_args = TrainingArguments(
        # output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        save_strategy='no',
        logging_strategy='epoch',
        eval_strategy='epoch',
        learning_rate=5e-5,
        report_to='none',
        fp16=False,
        use_cpu=True if device == 'cpu' else False,
    )

    data = load_data_for_triplet_train(filepath, dirpath, feature_extractor)
    data_collator = DataCollatorForTripletTrain(
        processor=feature_extractor,
    )

    print("Обучение с использованием триплетной потери")
    train(model, data, data_collator, training_args, output_dir)


def get_model_for_clf(model_dir, n_labels):
    model_id = 'facebook/hubert-base-ls960' if model_dir == '' else model_dir

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=n_labels,
    )

    is_local_file = model_dir == ''
    architecture = config.architectures[0]
    if architecture == 'HubertForTripletTrain':
        return HubertClassificationAfterTriplet.from_pretrained(
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


def classification_train(filepath, dirpath, output_dir, model_dir, n_labels, n_epochs, device, learning_rate, grad_accum_steps):
    model_id = 'facebook/hubert-base-ls960'
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    model = get_model_for_clf(model_dir, n_labels)

    if isinstance(model, HubertClassificationAfterTriplet):
        unfreeze_model_layers(model, 2)
    else:
        unfreeze_model_layers(model, 4 + model.config.num_hidden_layers * 16)

    if not torch.cuda.is_available() and device == 'gpu':
        print('CUDA недоступна, обучение будет происходить на CPU')
        device = 'cpu'

    training_args = TrainingArguments(
        # output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=grad_accum_steps,
        per_device_eval_batch_size=4,
        save_strategy='no',
        logging_strategy='epoch',
        eval_strategy='epoch',
        learning_rate=learning_rate,
        report_to='none',
        fp16=False,
        use_cpu=True if device == 'cpu' else False,
    )

    data_collator = DataCollatorForClassification(
        processor=feature_extractor,
    )

    print('Подготовка данных к обучению')

    data = load_data_for_clf_train(filepath, dirpath, feature_extractor)
    train(model, data, data_collator, training_args, output_dir, compute_metrics)


def train(model, data, data_collator, training_args, output_dir, compute_metrics_func=None):
    print('Начало обучения')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data['train'],
        eval_dataset=data['val'],
        compute_metrics=compute_metrics_func,
    )

    trainer.train()

    trainer.save_model(output_dir)

    print(f"Обученная модель сохранена в директорию '{output_dir}'")
