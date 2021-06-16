# Synthetic Data Generation
Algorithms for generating synthetic data. 

All algorithms are differentially private (DP) enabling the user to specify their desired privacy level.

We support the following synthesizers:
- MarginalSynthesizer: synthesize each column independently via DP marginal distributions.
- UniformSynthesizer: synthesize each column independently via uniform distributions. 
- ContingencySynthesizer: synthesize data via a DP contingency table of all attributes.
- PrivBayes (Zhang et al, 2017): synthesize data via a Bayesian network with DP conditional distributions.

## Install

Using `pip`:

```bash
pip install synthetic-data-generation
```

# Usage

```python
# import pandas and desired synthesis algorithm, e.g. PrivBayes
import pandas as pd
from synthesis.synthesizers import PrivBayes

# load data
df = pd.read_csv('examples/data/original/adult.csv')

# set desired privacy level (differential privacy) - default = 1
epsilon = 0.1

# instantiate and fit synthesizer
pb = PrivBayes(epsilon=epsilon)
pb.fit(df)

# Synthesize data
df_synth  = pb.sample()
```

Alternatively, check out the example notebooks under examples/tutorials.

