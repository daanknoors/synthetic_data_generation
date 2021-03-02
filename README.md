# Synthetic Data Generation
A repository consisting of algorithms for synthetic data generation. 

All algorithms are differentially private (DP) enabling the user to specify their desired privacy level.

We support the following synthesizers:
- MarginalSynthesizer: synthesize each column independently by the DP marginal distributions.
- ContingencySynthesizer: synthesize data via a DP contingency tables of all attributes.
- PrivBayes (Zhang et al, 2017): approximate the data through a Bayesian network with DP conditional distributions.

# Usage

```python
# import desired synthesis algorithm, e.g. PrivBayes
from synthesis.synthesizers.privbayes import PrivBayes
import pandas as pd

# load data
df = pd.read_csv('examples/data/input/adult_9c.csv')

# set desired privacy level (differential privacy) - default = 1
epsilon = 0.1

# instantiate and fit synthesizer
pb = PrivBayes(epsilon=epsilon)
pb.fit(df)

# Synthesize data
df_synth  = pb.sample()
```

Alternatively, check out the example notebooks under examples/tutorials.

