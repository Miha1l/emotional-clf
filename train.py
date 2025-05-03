from transformers import (
    HubertForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)

from models import (
    HubertForTripletTrain,
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

    if not torch.cuda.is_available() and device == 'GPU':
        print('CUDA недоступна, обучение будет происходить на CPU')
        device = 'CPU'

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        save_strategy='no',
        logging_steps=10,
        eval_strategy='epoch',
        learning_rate=5e-5,
        report_to='none',
        fp16=False,
        use_cpu=True if device == 'CPU' else False,
    )

    data = load_data_for_triplet_train(filepath, dirpath, feature_extractor)
    data_collator = DataCollatorForTripletTrain(
        processor=feature_extractor,
    )

    print("Обучение с использованием триплетной потери")
    train(model, data, data_collator, training_args, "")
    # unfreeze_model_layers(model, 2)
    # data = load_data_for_clf_train(filepath, dirpath, feature_extractor)
    # data_collator = DataCollatorForClassification(
    #     processor=feature_extractor,
    # )
    #
    # print("Обучение классификатора")
    # train(model, data, data_collator, training_args, output_dir)


def classification_train(filepath, dirpath, output_dir, n_labels, n_epochs, device, learning_rate, grad_accum_steps):
    model_id = "facebook/hubert-base-ls960"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=n_labels,
    )

    model = HubertForSequenceClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True
    )

    unfreeze_model_layers(model, 4 + config.num_hidden_layers * 16)

    if not torch.cuda.is_available() and device == 'gpu':
        print('CUDA недоступна, обучение будет происходить на CPU')
        device = 'CPU'

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        use_cpu=True if device == 'CPU' else False,
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
