## From qaskills.datasetprocessors.data.py from Sagnik's code
from dataclasses import dataclass
from typing import List

@dataclass
class Token:
    """
    token class
    """
    text: str
    char_start: int
    char_end: int
    sent: int

@dataclass
class Question:
    text: str
    tokens: List[Token]

@dataclass
class Context:
    text: str
    tokens: List[Token]

@dataclass
class RelevantSpan:
    text: str
    tokens: List[Token]
    contains_answer: bool

@dataclass
class QCA:
    """
    A holder class for question context and relevant spans (that may or may not be answers)
    """
    id: str
    question: Question
    context: Context
    relevant_spans: List[RelevantSpan]
    answer_texts_orig: List[str]
    answer_starts_orig: List[int]
    source: str