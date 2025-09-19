"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, List

#
# Implementation of a prelude of elementary functions.



# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    return -x


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return 1.0 / (1.0 + math.e**(-x)) if x >= 0 else math.e**x / (1.0 + math.e**x)


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return x if x > 0.0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    return 1 / x * d


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    return -(x**2) * d


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    return 0 if x < 0 else d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(l: Iterable, func: Callable) -> Iterable:
    r"Higher-order function that applies a given function to each element of an iterable"
    res = []
    for el in l:
        res.append(func(el))
    return res

def zipWith(l1: Iterable, l2: Iterable, func: Callable) -> Iterable:
    r"Higher-order function that applies a given function to each element of an iterable"
    res = []
    for el1, el2 in zip(l1, l2):
        res.append(func(el1, el2))
    return res

def reduce(l: Iterable, func: Callable) -> float:
    r"Higher-order function that applies a given function to each element of an iterable"
    res = 0
    first = True
    for el in l:
        if first:
            first = False
            res = el
        else:
            res = func(res, el)
    return res

def negList(l: List[float]):
    r"Negate all elements in a list using map"
    return map(l, neg)

def addLists(l1: List[float], l2: List[float]):
    r"Add corresponding elements from two lists using zipWith"
    return zipWith(l1, l2, add)

def sum(l: List[float]):
    r"Sum all elements in a list using reduce"
    return reduce(l, add)

def prod(l: List[float]):
    r"Calculate the product of all elements in a list using reduce"
    return reduce(l, mul)
