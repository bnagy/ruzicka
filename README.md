# Ružička: Authorship Verification in Python

## Introduction

## NOTE

This repository is based on the work (and paper) by [Mike Kestemont](https://github.com/mikekestemont/ruzicka)

The code here is:
- significantly stripped down
- updated for Python 3
- pylance clean and type hinted
- slightly optimised in a few places (it was already pretty fast)
- installable as a package

This was a *quick job* to fix it for my own use, and has not been carefully tested, but might be a useful starting point for someone.

## Installation

This is not really cooked enough for PiPi, so for now install straight off github.

`pip install git+https://github.com/bnagy/ruzicka@main#egg=ruzicka-imposters`

## Original Introduction

<img align="right" src="https://cloud.githubusercontent.com/assets/4376879/11402489/8703f80a-9398-11e5-8091-2b1ed5b2bb97.png" 
alt="IMAGE ALT TEXT HERE" height="240" border="10"/>
The code in this repository offers an implementation of a number of routines in authorship studies, with a focus on authorship verification. It is named after the inventor of the "minmax" measure (M. Ružička). The repository offers a generic implementation of two commonly used verification systems. The first system is an intrinsic verifier, depending on a first-order metric (O1), close to the one described in:

```
Potha, N. and E. Stamatatos. A Profile-based Method for Authorship Verification
In Proc. of the 8th Hellenic Conference on Artificial Intelligence
(SETN), LNCS, 8445, pp. 313-326, 2014.
```

The second system is an extrinsic verifier with second-order metrics (O2), based the General Imposters framework as described in:

```
M. Koppel and Y. Winter (2014), Determining if Two Documents are by the Same
Author, JASIST, 65(1): 178-187.
```

The package additionally offers a number of useful implementations of common vector space models and evaluation metrics. The code in this repository was used to produce the results in a paper ~~which is currently under submission~~.

## Quickstart

Most of the original Kestemont examples and bootstrap hypothesis testing framework has been removed. You can find that at the original repo. You can still follow the original [Quickstart notebook](code/Quickstart.ipynb) which covers the basic use of the vectorisation tools, and the O1 and O2 Verifiers.

## Dependencies

`setup.py` should take care of them for you, but for the record:

+ numpy
+ scipy
+ scikit-learn
+ numba




