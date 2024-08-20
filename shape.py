from dataclasses import dataclass


@dataclass
class Shape:
    height: int
    width: int
    n_colours: int = 3

    def as_tuple(self) -> tuple[int, int, int]:
        return self.height, self.width, self.n_colours

    @property
    def total_size(self) -> int:
        return self.height * self.width * self.n_colours
