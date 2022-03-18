"""Functions to help perform an experimental run of synthesis"""
from itertools import product
from pathlib import Path

from synthesis.synthesizers.privbayes import PrivBayes
from synthesis.synthesizers.marginal import MarginalSynthesizer

def synthesis_experiment(X, X_name, synthesizers=None, epsilon=None, n_records_synth=None, path=None, verbose=2):
    if synthesizers is None:
        synthesizers = [MarginalSynthesizer, PrivBayes]

    if not isinstance(path, Path):
        path = Path(path)
    model_path = path / 'output_model'
    data_path = path / 'output_data'

    if epsilon is None:
        epsilon = [0.01, 0.1, 1]

    for sts, e in product(synthesizers, epsilon):
        synthesizer = sts(epsilon=e, verbose=verbose)
        if isinstance(synthesizer, MarginalSynthesizer):
            synthesizer.verbose = 0

        synthesizer.fit(X)
        # Xs = synthesizer.sample()
        # synthesizer.save(model_path)
        print('Synthesis complete for: {}'.format(sts))

    return synthesizer