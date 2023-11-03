# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for saving provenances in tsdate.
"""
import json

import numpy as np
import pytest
import utility_functions

import tsdate
from tsdate import provenance


class TestProvenance:
    def test_bad_get_dict(self):
        with pytest.raises(ValueError, match="cannot be None"):
            provenance.get_provenance_dict(None)

    def test_date_cmd_recorded(self):
        ts = utility_functions.single_tree_ts_n2()
        dated_ts = tsdate.date(ts, population_size=1, mutation_rate=None)
        rec = json.loads(dated_ts.provenance(-1).record)
        assert rec["software"]["name"] == "tsdate"
        assert rec["parameters"]["command"] == "date"

    def test_date_params_recorded(self):
        ts = utility_functions.single_tree_ts_n2()
        mu = 0.123
        Ne = 9
        dated_ts = tsdate.date(
            ts, population_size=Ne, mutation_rate=mu, method="maximization"
        )
        rec = json.loads(dated_ts.provenance(-1).record)
        assert np.isclose(rec["parameters"]["mutation_rate"], mu)
        assert np.isclose(rec["parameters"]["population_size"], Ne)
        assert rec["parameters"]["method"] == "maximization"

    @pytest.mark.parametrize(
        "popdict",
        [
            {"population_size": [1, 2, 3], "time_breaks": [1, 1.2]},
            {"population_size": [123]},
        ],
    )
    def test_date_popsizehist_recorded(self, popdict):
        ts = utility_functions.single_tree_ts_n2()
        mu = 0.123
        for use_class in (False, True):
            if use_class:
                popsize = tsdate.demography.PopulationSizeHistory(**popdict)
            else:
                popsize = popdict
            dated_ts = tsdate.date(ts, population_size=popsize, mutation_rate=mu)
            rec = json.loads(dated_ts.provenance(-1).record)
            assert np.isclose(rec["parameters"]["mutation_rate"], mu)
            assert "population_size" in rec["parameters"]
            popsz = rec["parameters"]["population_size"]
            assert len(popsz) == len(popdict)
            for param, val in popdict.items():
                assert np.all(np.isclose(val, popsz[param]))

    def test_preprocess_cmd_recorded(self):
        ts = utility_functions.ts_w_data_desert(40, 60, 100)
        preprocessed_ts = tsdate.preprocess_ts(ts)
        rec = json.loads(preprocessed_ts.provenance(-1).record)
        assert rec["software"]["name"] == "tsdate"
        assert rec["parameters"]["command"] == "preprocess_ts"

    def test_preprocess_defaults_recorded(self):
        ts = utility_functions.ts_w_data_desert(40, 60, 100)
        preprocessed_ts = tsdate.preprocess_ts(ts)
        rec = json.loads(preprocessed_ts.provenance(-1).record)
        assert rec["parameters"]["remove_telomeres"]
        assert rec["parameters"]["minimum_gap"] == 1000000
        assert rec["parameters"]["delete_intervals"] == []

    def test_preprocess_interval_recorded(self):
        ts = utility_functions.ts_w_data_desert(40, 60, 100)
        preprocessed_ts = tsdate.preprocess_ts(
            ts, minimum_gap=20, remove_telomeres=False
        )
        rec = json.loads(preprocessed_ts.provenance(-1).record)
        print(rec)
        assert rec["parameters"]["minimum_gap"] == 20
        assert rec["parameters"]["remove_telomeres"] is not None
        assert not rec["parameters"]["remove_telomeres"]
        deleted_intervals = rec["parameters"]["delete_intervals"]
        assert len(deleted_intervals) == 1
        assert deleted_intervals[0][0] < deleted_intervals[0][1]
        assert 40 < deleted_intervals[0][0] < 60
        assert 40 < deleted_intervals[0][1] < 60
