# Transformers

Reversible transformers:
1. pre-process original data to format that fits the synthesizer requirements;
2. post-process synthetic data to reverse all pre-processing steps to resemble the original domain.

Supported:
- discretization: generalizing columns to reduce dimensionality
- sequence: categorize date sequences

Roadmap:
- merge: merge datasets based on identifiers
- deidentification: remove rows and columns, and values that are too sensitive.
- flatten: flatten sequential data