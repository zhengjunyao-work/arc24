from abc import ABC, abstractmethod
import numpy as np


def create_grid_encoder(encoder_name):
    """
    This is a security risk, but I'm the only user of the library
    Otherwise I will need to write and maintain dictionary with the encoder names
    It allows me to use plain text in the configuration to specify the encoder

    Examples of encoders:

    GridCodeBlockEncoder(MinimalGridEncoder())
    GridCodeBlockEncoder(GridWithSeparationEncoder('|'))
    """
    grid_encoder = eval(encoder_name)
    if isinstance(grid_encoder, GridEncoder):
        print(f'Created {encoder_name} as grid encoder')
        return grid_encoder
    else:
        raise ValueError(f'{encoder_name} is not a GridEncoder subclass')


class GridEncoder(ABC):
    @abstractmethod
    def to_text(self, grid):
        pass

    @abstractmethod
    def to_grid(self, text):
        pass


class MinimalGridEncoder(GridEncoder):
    @staticmethod
    def to_text(grid):
        text = '\n'.join([''.join([str(x) for x in line]) for line in grid])
        return text

    @staticmethod
    def to_grid(text):
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line] for line in lines]
        return grid


class GridWithSeparationEncoder(GridEncoder):
    def __init__(self, split_symbol):
        self.split_symbol = split_symbol

    def to_text(self, grid):
        text = '\n'.join([self.split_symbol.join([str(x) for x in line]) for line in grid])
        return text

    def to_grid(self, text):
        lines = text.strip().splitlines()
        grid = [[int(x) for x in line.split(self.split_symbol)] for line in lines]
        return grid


class GridCodeBlockEncoder(GridEncoder):
    def __init__(self, base_encoder):
        self.encoder = base_encoder

    def to_text(self, grid):
        text = f'```grid\n{self.encoder.to_text(grid)}\n```'
        return text

    def to_grid(self, text):
        grid_text = text.split('```grid\n')[1].split('\n```')[0]
        grid = self.encoder.to_grid(grid_text)
        return grid


class RepeatNumberEncoder(GridEncoder):
    def __init__(self, n=3):
        self.n = n

    def to_text(self, grid):
        text = '\n'.join([''.join([str(x)*self.n for x in line]) for line in grid])
        return text

    def to_grid(self, text):
        lines = text.strip().splitlines()
        #TODO: make something more robust
        grid = [[int(x) for x in line[::self.n]] for line in lines]
        return grid


def test_translator(translator):
    sample_grid = np.eye(3, dtype=int).tolist()
    assert sample_grid == translator.to_grid(translator.to_text(sample_grid))
    print(type(translator).__name__)
    print(translator.to_text(sample_grid) + '\n')


if __name__ == '__main__':
    test_translator(MinimalGridEncoder())
    test_translator(GridWithSeparationEncoder('|'))
    test_translator(GridCodeBlockEncoder(MinimalGridEncoder()))
    test_translator(GridCodeBlockEncoder(GridWithSeparationEncoder('|')))
    test_translator(RepeatNumberEncoder())
    test_translator(RepeatNumberEncoder(2))
