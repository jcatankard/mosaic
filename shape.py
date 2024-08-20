from dataclasses import dataclass


@dataclass
class Shape:
    height: int
    width: int
    n_colours: int = 3

    @property
    def total_size(self) -> int:
        return self.height * self.width * self.n_colours
