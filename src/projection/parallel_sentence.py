from utils.coo_matrix_nocheck import CooMatrix
from typing import List


class SourceSentence:
    def __init__(self, weights: CooMatrix, pos: List[str], forms: List[str], language: str, alignments: List[tuple]):
        self.weights = weights
        self.pos = pos
        self.forms = forms
        self.language = language
        self.alignments = alignments


class ParallelSentence:
    def __init__(self, target: List[str], sources: List[SourceSentence], gold=None, silver=None):
        self.sources = sources
        self.target = target
        self.gold = gold
        self.silver = silver
