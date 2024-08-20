from dataclasses import dataclass


@dataclass
class Shape:
    height: int
    width: int
    n_colours: int = 3
