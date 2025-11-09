from dataclasses import dataclass

@dataclass
class TrainingData:
    inputs: list[float]
    expected: list[float]