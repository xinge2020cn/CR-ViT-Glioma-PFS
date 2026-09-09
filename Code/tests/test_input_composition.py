"""Constructed fixtures test validation logic, not clinical findings."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd
spec=importlib.util.spec_from_file_location('input_validation',Path(__file__).resolve().parents[1]/'00_validate_inputs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class CompositionTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(dict(enhancing_proportion=['Moderate (>33-67%)'],necrotic_proportion=['Mild (>5-33%)'],enhancing_fraction_core=[.6],necrotic_fraction_core=[.2],other_fraction_core=[.2],tumor_volume_cm3=[50.],enhancing_volume_cm3=[30.],necrotic_volume_cm3=[10.],other_core_volume_cm3=[10.]))
    def test_valid(self):self.assertTrue(all(module.composition_checks(self.fixture()).values()))
    def test_impossible_bins(self):
        d=self.fixture();d.loc[0,'enhancing_proportion']='Extensive (>67%)';d.loc[0,'necrotic_proportion']='Extensive (>67%)'
        self.assertFalse(module.composition_checks(d)['Joint composition categories are feasible'])
    def test_wrong_sum(self):
        d=self.fixture();d.loc[0,'other_fraction_core']=.5
        self.assertFalse(module.composition_checks(d)['Core fractions are finite and sum to one'])
    def test_stale_category(self):
        d=self.fixture();d.loc[0,'enhancing_proportion']='None/minimal (<=5%)'
        self.assertFalse(module.composition_checks(d)['Category matches enhancing_fraction_core'])
    def test_stale_volume(self):
        d=self.fixture();d.loc[0,'necrotic_volume_cm3']=40.
        self.assertFalse(module.composition_checks(d)['Volume matches necrotic_fraction_core'])
    def test_boundary(self):
        d=self.fixture();d.loc[0,['enhancing_fraction_core','necrotic_fraction_core','other_fraction_core']]=[.67,.33,0]
        d.loc[0,['enhancing_volume_cm3','necrotic_volume_cm3','other_core_volume_cm3']]=[33.5,16.5,0]
        self.assertTrue(all(module.composition_checks(d).values()))
if __name__=='__main__':unittest.main()
