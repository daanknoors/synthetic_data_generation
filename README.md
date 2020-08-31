# Synthetic Data Generation
A repository consisting of algorithms for synthetic data generation. 

All algorithms are differentially private (DP) enabling the user to specify their desired privacy guarantee, i.e. epsilon.

Currently supporting the following algorithms:
- MarginalSynthesizer: synthesize each column independently by the DP marginal distributions.
- PrivBayes (Zhang et al, 2017): approximate the data through a Bayesian network with DP conditional distributions.
- MetaSynthesizer: combines discretization procedures with PrivBayes to handle data of varying dimensionalities.

We chose a modular approach, thus the discretization procedures in MetaSynthesizer can also be run separately. We support the following:
- GeneralizeContinuous: bins continuous values in ranges.
- GeneralizeDataSequence: converts multiple sequential date columns to categorical ranges.
- GeneralizeCategorical: generalizes high-cardinality categorical columns into groups.

These discretization modules can be run prior to synthesis on the original data. This ensures that the input dimensions will fit into memory and the synthesis will run faster. This is important when using algorithms that try to capture patterns between features, e.g. PrivBayes. Additionally, the output synthetic dataset can be inverse-transformed to again obtain the original representation. MetaSynthesizer can do all of these procedures automatically.
	
# Usage
Usage follows similar approach to scikit-learn transformers. 

```python
# import desired synthesis algorithm, e.g. PrivBayes
from synthesis.bayes_synthesis import PrivBayes
import pandas as pd

# load data
df = pd.read_csv('examples/data/input/adult.csv')

# set desired privacy level (differential privacy) - default = 1
epsilon = 0.1

# instantiate and fit synthesizer
pb = PrivBayes(epsilon=epsilon)
pb.fit(df)

# Synthesize data
df_synth  = pb.transform(df)
```

Alternatively, check out the example notebooks under examples/tutorials.

