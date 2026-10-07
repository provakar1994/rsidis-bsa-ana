import numpy as np
import pandas as pd
import pytest

from plot_nuclear_dependence import _select_kinematics


def test_select_x_then_z_without_mixing_adjacent_settings():
    df = pd.DataFrame({'x': [0.25, 0.25, 0.25, 0.44],
                       'z': [0.5, 0.52, 0.67, 0.5]})
    selected = _select_kinematics(df, 0.25, [0.5, 0.67, 0.5])
    assert selected.x.tolist() == [0.25, 0.25]
    assert selected.z.tolist() == [0.5, 0.67]
    assert len(df) == 4


def test_multiple_x_requires_explicit_selection():
    with pytest.raises(ValueError, match='Specify --x'):
        _select_kinematics(pd.DataFrame({'x': [0.25, 0.44], 'z': [0.5, 0.5]}), None, None)


def test_single_x_can_be_inferred():
    df = pd.DataFrame({'x': [0.25], 'z': [0.5]})
    pd.testing.assert_frame_equal(_select_kinematics(df, None, None), df)


@pytest.mark.parametrize('x,z', [(0.44, None), (0.25, [0.67]), (np.nan, None)])
def test_missing_selection_reports_error(x, z):
    with pytest.raises(ValueError, match='No data'):
        _select_kinematics(pd.DataFrame({'x': [0.25], 'z': [0.5]}), x, z)
