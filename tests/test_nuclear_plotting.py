import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plot_nuclear_dependence import (
    _apply_real_z, _particle_average, _fig_grid, _fig_zavg_diff,
    _compute_diff, _compute_diff_pt, _z_ivw_average, _to_flu_fuu,
)


def sample():
    return pd.DataFrame([
        dict(target=t, particle=p, histogram='pt0', hmin=0., hmax=.2,
             bin_center=.1, z=z, x=.25, q2=3.3, asym=a + shift, asym_err=e)
        for t, shift in [('LH2', 0.), ('C', .2)]
        for z in [.5, .67]
        for p, a, e in [('pi+', .1, .02), ('pi-', .3, .04)]
    ])


def test_inverse_variance_average_and_missing_charge():
    df = sample()
    avg = _particle_average(df, 'z')
    np.testing.assert_allclose(avg[avg.target == 'LH2'].asym, .14)
    np.testing.assert_allclose(avg.asym_err, .02 / np.sqrt(1.25))
    assert _particle_average(df[df.particle == 'pi+'], 'z').empty
    df.loc[df.particle == 'pi-', 'asym_err'] = 0
    assert _particle_average(df, 'z').empty


def test_real_z_preserves_selection_keys_and_supports_specific_csv(tmp_path):
    df = sample()
    path = tmp_path / 'z.csv'
    pd.DataFrame([dict(z=.5, real_z=.53, target='C', particle='pi+')]).to_csv(path, index=False)
    result = _apply_real_z(df, ['0.5=0.51'], str(path))
    pd.testing.assert_series_equal(result.z, df.z)
    assert result.query("target == 'C' and particle == 'pi+' and z == .5").plot_z.iloc[0] == .53
    assert result.query("target == 'LH2' and z == .5").plot_z.tolist() == [.51, .51]
    assert result.query('z == .67').plot_z.eq(.67).all()
    avg = _particle_average(result, 'z')
    assert avg.query("target == 'C' and z == .5").plot_z.iloc[0] == pytest.approx(.52)
    with pytest.raises(ValueError, match='NOMINAL=REAL'):
        _apply_real_z(df, ['oops'])
    with pytest.raises(ValueError, match='0 < real'):
        _apply_real_z(df, ['0.5=nan'])


@pytest.mark.parametrize('flu', [False, True])
@pytest.mark.parametrize('difference', [False, True])
def test_grid_offsets_average_and_single_charge(flu, difference):
    df = _apply_real_z(sample(), ['0.5=0.51'])
    if flu:
        df = _to_flu_fuu(df, .59)
    if difference:
        df = _compute_diff(df, 'LH2')
    bins = df[['histogram', 'hmin', 'hmax', 'bin_center']].drop_duplicates()
    fig = _fig_grid(df, ['C'], bins, 'Test')
    ax = fig.axes[0]
    series = ax.containers
    assert len(series) == 3
    xp = np.asarray(series[0].lines[0].get_xdata(), dtype=float)
    xm = np.asarray(series[1].lines[0].get_xdata(), dtype=float)
    xa = np.asarray(series[2].lines[0].get_xdata(), dtype=float)
    assert np.all(xp < xa) and np.all(xa < xm)
    np.testing.assert_allclose(xa, [.51, .67])
    assert 'weighted average' in ax.get_legend().get_texts()[2].get_text()
    plt.close(fig)
    fig = _fig_grid(df[df.particle == 'pi-'], ['C'], bins, 'Test')
    assert len(fig.axes[0].containers) == 1
    np.testing.assert_allclose(np.asarray(fig.axes[0].containers[0].lines[0].get_xdata(), dtype=float), [.51, .67])
    assert len(fig.axes[0].get_legend().get_texts()) == 1
    plt.close(fig)


def test_z_averaged_difference_has_centered_average():
    avg = _z_ivw_average(sample(), [.5, .67])
    diff = _compute_diff_pt(avg, 'LH2')
    fig = _fig_zavg_diff(diff, ['C'], 'Test')
    series = fig.axes[0].containers
    assert len(series) == 3
    assert series[0].lines[0].get_xdata()[0] < .1
    assert series[1].lines[0].get_xdata()[0] > .1
    assert series[2].lines[0].get_xdata()[0] == .1
    assert series[2].lines[0].get_ydata()[0] == pytest.approx(.2)
    plt.close(fig)


def test_z_average_page_title_uses_real_coordinates():
    from plot_nuclear_dependence import _emit_zavg_diff_pages, _QTY_A

    class CapturePDF:
        def savefig(self, fig, **kwargs):
            self.title = fig._suptitle.get_text()

    pdf = CapturePDF()
    df = _apply_real_z(sample(), ['0.5=0.513', '0.67=0.681'])
    assert _emit_zavg_diff_pages(pdf, df, ['C'], [.5, .67], 'LH2', _QTY_A, None) == 1
    assert '[z-avg: {0.513, 0.681}]' in pdf.title


@pytest.mark.parametrize('particle', ['pi+', 'pi-', 'both'])
def test_z_average_comparison_includes_reference_and_target_values(particle):
    from plot_nuclear_dependence import _emit_zavg_comparison_page, _QTY_A

    class CapturePDF:
        def savefig(self, fig, **kwargs):
            self.title = fig._suptitle.get_text()
            self.axes = fig.axes

    df = _apply_real_z(sample(), ['0.5=0.513', '0.67=0.681'])
    if particle != 'both':
        df = df[df.particle == particle]
    pdf = CapturePDF()
    assert _emit_zavg_comparison_page(pdf, df, ['LH2', 'C'], [.5, .67],
                                      _QTY_A, (-.1, .6)) == 1
    assert '[z-avg: {0.513, 0.681}]' in pdf.title
    assert len(pdf.axes) == 2
    assert len(pdf.axes[0].containers) == (3 if particle == 'both' else 1)
    expected = {'pi+': .1, 'pi-': .3, 'both': .14}[particle]
    assert pdf.axes[0].containers[-1].lines[0].get_ydata()[0] == pytest.approx(expected)
    assert pdf.axes[1].containers[-1].lines[0].get_ydata()[0] == pytest.approx(expected + .2)
    assert pdf.axes[0].get_ylim() == (-.1, .6)
