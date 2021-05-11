"""Functions for comparing descriptive statistics"""

from dython.nominal import compute_associations

from synthesis.evaluation._base import BaseStatistic

class Associations(BaseStatistic):

    def __init__(self, theil_u=True, nominal_columns='auto'):
        self.theil_u = True
        self.nominal_columns = 'auto'

    def describe(self, data):
        return compute_associations(data, theil_u=self.theil_u, nominal_columns=self.nominal_columns)

