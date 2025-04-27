import argparse
import pathlib

import pandas as pd

from inference import make_predicts
from utils import args_checkers as checker


def predict_mode(args):
    predictions = make_predicts(args.dir)
    # for predict in predictions:
    #     print(predict)

    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(args.output)

    print(f'Результаты сохранены в файл {args.output}')


def train_mode(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Приложение для интеллектуального анализа эмоциональной речи')
    subparsers = parser.add_subparsers(title='Режимы работы',
                                       description='',
                                       help='Для вызова справки по нужному режиму вызовите {predict/train} -h/--help')

    predict_parser = subparsers.add_parser('predict', help='Классификация аудио')
    predict_parser.add_argument('-d', '--dir', type=checker.dirpath_checker, required=True,
                                help='Путь до папки с аудиофайлами .wav')
    predict_parser.add_argument('-o', '--output', type=checker.filepath_checker, default='results.csv',
                                help='Имя csv-файла для записи результатов классификации (По умолчанию: %(default)s)')
    predict_parser.set_defaults(func=predict_mode)

    train_parser = subparsers.add_parser('train', help='Дообучение модели')
    train_parser.add_argument('-d', '--dir', type=pathlib.Path, required=True,
                              help='Путь до папки с аудиофайлами .wav')
    train_parser.add_argument('--epochs', type=int, default=5,
                              help='Количество эпох обучения (По умолчанию: %(default)s)')
    train_parser.add_argument('--arch', choices=['hubert'], default='hubert',
                              help='Архитектура модели (По умолчанию: %(default)s)')
    train_parser.add_argument('--loss', choices=['CrossEntropy', 'Triplet'], default='CrossEntropy',
                              help='Функция ошибки (По умолчанию: %(default)s)')
    train_parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                              help='Устройство, на котором будет выполняться обучение (По умолчанию: %(default)s)')

    args = parser.parse_args()
    if not vars(args):
        parser.print_help()
    else:
        args.func(args)
