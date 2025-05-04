import argparse
import pathlib

import pandas as pd

from inference import make_predicts
from train import classification_train, triplet_train
from test import test_model
from utils import args_checkers as checker


def predict_mode(args):
    predictions = make_predicts(args.dir, args.model)
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(args.output)

    print(f'Результаты сохранены в файл {args.output}')


def train_mode(args):
    if args.loss == 'CrossEntropy':
        classification_train(args.file, args.dir, args.output, args.model, args.n_classes,
                             args.epochs, args.device, args.lr, args.grad_accum_steps)
    elif args.loss == 'TripletLoss':
        triplet_train(args.file, args.dir, args.output, args.n_epochs, args.device)


def test_mode(args):
    test_model(args.file, args.dir, args.model, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Приложение для интеллектуального анализа эмоциональной речи')
    subparsers = parser.add_subparsers(title='Режимы работы',
                                       description='',
                                       help='Для вызова справки по нужному режиму вызовите {predict/train} -h/--help')

    predict_parser = subparsers.add_parser('predict', help='Классификация аудио')
    predict_parser.add_argument('-d', '--dir', type=checker.dirpath_checker, required=True,
                                help='Путь до папки с аудиофайлами .wav')
    predict_parser.add_argument('-m', '--model', type=checker.dirpath_checker, required=True,
                                help='Путь до модели классификации')
    predict_parser.add_argument('-o', '--output', type=checker.filepath_checker, default='results.csv',
                                help='Имя csv-файла для записи результатов классификации (По умолчанию: %(default)s)')
    predict_parser.add_argument('-n', '--n_classes', choices=[2], default=2,
                                help='Количество классов эмоций (По умолчанию: %(default)s)')
    predict_parser.set_defaults(func=predict_mode)

    train_parser = subparsers.add_parser('train', help='Дообучение модели')
    train_parser.add_argument('-d', '--dir', type=checker.dirpath_checker, required=True,
                              help='Путь до папки с аудиофайлами .wav')
    train_parser.add_argument('-f', '--file', type=checker.filepath_checker, required=True,
                              help='Путь до .csv файла с разметкой файлов по эмоциям')
    train_parser.add_argument('-m', '--model', type=checker.filepath_checker, default='',
                              help='Путь до базовой модели (По умолчанию используется facebook/hubert-base-ls960')
    train_parser.add_argument('-n', '--n_classes', choices=[2], default=2,
                              help='Количество классов эмоций (По умолчанию: %(default)s)')
    train_parser.add_argument('-o', '--output', type=checker.filepath_checker, default='hubert-ft',
                              help='Имя папки для записи модели (По умолчанию: %(default)s)')
    train_parser.add_argument('-e', '--epochs', type=int, default=5,
                              help='Количество эпох обучения (По умолчанию: %(default)s)')
    train_parser.add_argument('--grad_accum_steps', type=int, default=2,
                              help='Количество шагов накопления градиента (По умолчанию: %(default)s)')
    train_parser.add_argument('--lr', type=float, default=5e-5,
                              help='Темп обучения (По умолчанию: %(default)s)')
    train_parser.add_argument('--loss', choices=['CrossEntropy', 'Triplet'], default='CrossEntropy',
                              help='Функция ошибки (По умолчанию: %(default)s)')
    train_parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu',
                              help='Устройство, на котором будет выполняться обучение (По умолчанию: %(default)s)')
    train_parser.set_defaults(func=train_mode)

    test_parser = subparsers.add_parser('test', help='Тестирование модели')
    test_parser.add_argument('-d', '--dir', type=checker.dirpath_checker, required=True,
                             help='Путь до папки с аудиофайлами .wav')
    test_parser.add_argument('-f', '--file', type=checker.filepath_checker, required=True,
                             help='Путь до .csv файла с разметкой файлов по эмоциям')
    test_parser.add_argument('-m', '--model', type=checker.dirpath_checker, required=True,
                             help='Путь до модели классификации')
    test_parser.add_argument('-o', '--output', type=checker.filepath_checker, default='metrics.json',
                             help='Имя json-файла для сохранения метрик (По умолчанию: %(default)s)')
    test_parser.set_defaults(func=test_mode)

    args = parser.parse_args()
    if not vars(args):
        parser.print_help()
    else:
        args.func(args)
