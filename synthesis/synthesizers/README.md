# Synthesizers

All synthesizers in this repo are differentially private.

Common hyperparameters:
- epsilon: privacy level
- verbose: text displayed

After fitting the synthesizer it has at least the following attributes:
- model_: differentially private model of the original data
- n_records_fit_: number of records in original data (used if no value is specified in sampling)
- columns_: columns in original data


