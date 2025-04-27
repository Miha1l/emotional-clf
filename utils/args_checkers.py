from argparse import ArgumentTypeError

import os
import pathlib


def dirpath_checker(path):
    if os.path.exists(path):
        return path
    else:
        raise ArgumentTypeError(f'Директория {path} не найдена')


def filepath_checker(path):
    if pathlib.Path(path).parent.absolute().exists():
        return path
    else:
        raise ArgumentTypeError(f'Путь до файла {path} не найден')
