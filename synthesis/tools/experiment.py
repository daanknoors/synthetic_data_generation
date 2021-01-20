"""Functions to help perform an experimental run of synthesis"""
import warnings
from sys import maxsize
from numbers import Real
import numpy as np
import pandas as pd
import os
from numpy.core import multiarray as mu
from numpy.core import umath as um
from itertools import product
from pathlib import Path
from thomas.core import CPT

from synthesis.bayes_synthesis import PrivBayes
from synthesis.hist_synthesis import MarginalSynthesizer
from synthesis.meta_synthesis import MetaSynthesizer

def synthesis_experiment(X, X_name, synthesizers=None, epsilon=None, n_records_synth=None, path=None, verbose=2):
    if synthesizers is None:
        synthesizers = [MarginalSynthesizer, PrivBayes, MetaSynthesizer]

    if not isinstance(path, Path):
        path = Path(path)
    model_path = path / 'output_model'
    data_path = path / 'output_data'

    if epsilon is None:
        epsilon = [0.01, 0.1, 1]

    for sts, e in product(synthesizers, epsilon):
        synthesizer = sts(epsilon=e, n_records_synth=n_records_synth, verbose=verbose)
        if isinstance(synthesizer, MarginalSynthesizer):
            synthesizer.verbose = 0

        synthesizer.fit(X)
        Xs = synthesizer.transform(X)
        synthesizer.write_csv(Xs, X_name, data_path)
        synthesizer.write_class(X_name, model_path)
        print('Synthesis complete for: {}'.format(sts))

