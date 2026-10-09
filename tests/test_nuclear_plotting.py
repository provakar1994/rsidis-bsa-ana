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


def systematic_table(tmp_path, df=None):
    from plot_nuclear_dependence import _load_systematics
    df = sample() if df is None else df
    rows = []
    for target in df.target.unique():
        for particle in ['pi+', 'pi-', 'both']:
            for view, zvals in [('z', [.5, .67]), ('zavg', [None])]:
                for z in zvals:
                    for ref in ['', 'LH2']:
                        rows.append(dict(process='sidis', x=.25, target=target,
                                         reference=ref, particle=particle, view=view, z=z,
                                         histogram='pt0', beam_pol_pct=3., excl_pct=4.,
                                         delta_pct=0., rho_pct=0.))
    path = tmp_path / 'sys.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return _load_systematics(str(path)), path


@pytest.mark.parametrize('particle', ['pi+', 'pi-', 'both'])
@pytest.mark.parametrize('difference', [False, True])
@pytest.mark.parametrize('axis', ['z', 'bin_center'])
def test_systematic_percentages_apply_to_final_value(tmp_path, particle, difference, axis):
    from plot_nuclear_dependence import _systematic_points
    table, _ = systematic_table(tmp_path)
    df = sample()
    if particle != 'both':
        df = df[df.particle == particle]
    if axis == 'bin_center':
        df = _z_ivw_average(df, [.5, .67])
    if difference:
        df = (_compute_diff(df, 'LH2') if axis == 'z'
              else _compute_diff_pt(df, 'LH2'))
    targets = ['C'] if difference else ['LH2', 'C']
    points = _systematic_points(df, targets, axis, table, 'LH2' if difference else None)
    assert points.particle.eq(particle).all()
    np.testing.assert_allclose(points.sigma_sys, points.asym.abs() * .05)
    expected = .2 if difference else {'pi+': .1, 'pi-': .3, 'both': .14}[particle]
    point = points.iloc[0] if difference else points[points.target == "LH2"].iloc[0]
    assert point.sigma_sys == pytest.approx(expected * .05)
    scaled = _systematic_points(_to_flu_fuu(df, .59), targets, axis, table,
                                'LH2' if difference else None)
    np.testing.assert_allclose(scaled.sigma_sys, points.sigma_sys / np.sqrt(2 * .59 * .41))


def test_systematic_absolute_value_and_zero_average(tmp_path):
    from plot_nuclear_dependence import _systematic_points
    table, _ = systematic_table(tmp_path)
    df = sample()
    df.loc[df.particle == 'pi+', 'asym'] = -.1
    df.loc[df.particle == 'pi-', 'asym'] = .4
    points = _systematic_points(df, ['C'], 'z', table)
    np.testing.assert_allclose(points.sigma_sys, 0., atol=1e-15)
    points = _systematic_points(df[df.particle == 'pi+'], ['C'], 'z', table)
    np.testing.assert_allclose(points.sigma_sys, .005)


@pytest.mark.parametrize('mutation,message', [
    ('duplicate', 'duplicate'), ('negative', 'nonnegative'),
    ('nan', 'finite'), ('missing_column', 'missing columns'),
    ('bad_view', 'view must'), ('bad_zavg', 'empty z'),
])
def test_systematic_csv_validation(tmp_path, mutation, message):
    from plot_nuclear_dependence import _load_systematics
    _, path = systematic_table(tmp_path)
    rows = pd.read_csv(path)
    if mutation == 'duplicate':
        rows = pd.concat([rows, rows.iloc[:1]])
    elif mutation == 'negative':
        rows.loc[0, 'excl_pct'] = -1
    elif mutation == 'nan':
        rows.loc[0, 'rho_pct'] = np.nan
    elif mutation == 'missing_column':
        rows = rows.drop(columns='beam_pol_pct')
    elif mutation == 'bad_view':
        rows.loc[0, 'view'] = 'other'
    else:
        rows.loc[rows.view == 'zavg', 'z'] = .5
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        _load_systematics(str(path))


def test_missing_both_or_difference_rows_never_fall_back(tmp_path):
    from plot_nuclear_dependence import _systematic_points
    table, _ = systematic_table(tmp_path)
    with pytest.raises(ValueError, match='missing .*plotted point'):
        _systematic_points(sample(), ['C'], 'z', table[table.particle != 'both'])
    with pytest.raises(ValueError, match='missing .*plotted point'):
        _systematic_points(_compute_diff(sample(), 'LH2'), ['C'], 'z',
                           table[table.reference != 'LH2'], 'LH2')


def test_smooth_systematic_curve_preserves_heights_without_overshoot():
    from plot_nuclear_dependence import _systematic_curve
    points = pd.DataFrame(dict(plot_z=[.51, .68, .89], sigma_sys=[.01, .03, .02]))
    x, height = _systematic_curve(points, 'z')
    assert x[0] == .51 and x[-1] == .89
    assert height.min() >= .01 and height.max() <= .03
    for xi, yi in zip(points.plot_z, points.sigma_sys):
        assert height[x == xi][0] == pytest.approx(yi)


@pytest.mark.parametrize('axis', ['z', 'bin_center'])
def test_systematic_band_has_flat_baseline_and_keeps_limits(tmp_path, axis):
    table, _ = systematic_table(tmp_path)
    df = _apply_real_z(sample(), ['0.5=0.513', '0.67=0.681'])
    if axis == 'z':
        bins = df[['histogram', 'hmin', 'hmax', 'bin_center']].drop_duplicates()
        fig = _fig_grid(df, ['C'], bins, 'Test', ylim=(-.1, .6),
                        systematics=table, sys_y=-.08)
    else:
        fig = _fig_zavg_diff(_z_ivw_average(df, [.5, .67]), ['C'], 'Test',
                             ylim=(-.1, .6), systematics=table, sys_y=-.08)
    ax = fig.axes[0]
    band = [c for c in ax.collections if c.get_label() == 'Systematic uncertainty (+1σ)'][0]
    vertices = band.get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(-.08)
    assert vertices[:, 1].max() > -.08
    assert ax.get_ylim() == (-.1, .6)
    assert any('Systematic' in text.get_text() for text in ax.get_legend().get_texts())
    plt.close(fig)


@pytest.mark.parametrize('flags,message', [
    (['--sys'], '--sys requires --sys-csv'),
    (['--sys', '--sys-csv', '/tmp/ssa-systematics-file-does-not-exist.csv'], 'No such file'),
    (['--sys-csv', 'unused.csv'], 'require --sys'),
])
def test_systematic_cli_requires_existing_csv(monkeypatch, capsys, flags, message):
    import sys
    from plot_nuclear_dependence import main
    monkeypatch.setattr(sys, 'argv', ['plot_nuclear_dependence.py', *flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_entries_fail_before_output_is_created(tmp_path, monkeypatch, capsys):
    import sys
    import plot_nuclear_dependence as plotting
    _, path = systematic_table(tmp_path)
    table = pd.read_csv(path)
    table = table[~((table.reference == 'LH2') & (table.particle == 'both'))]
    table.to_csv(path, index=False)
    out = tmp_path / 'should-not-exist.pdf'
    monkeypatch.setattr(plotting, '_load_all', lambda *args: sample())
    monkeypatch.setattr(sys, 'argv', ['plot_nuclear_dependence.py', '--x', '.25',
                                     '--sys', '--sys-csv', str(path), '--output', str(out)])
    with pytest.raises(SystemExit) as exc:
        plotting.main()
    assert exc.value.code == 2
    assert 'missing' in capsys.readouterr().err
    assert not out.exists()


def test_automatic_summary_contains_final_z_averages_and_charge_averages():
    from plot_nuclear_dependence import _plot_summary
    df = _apply_real_z(sample(), ['0.5=0.53', '0.67=0.71'])
    summary = _plot_summary(df, ['LH2', 'C'], 'LH2', [.5, .67], .59)
    assert len(summary) == 54
    assert set(summary.observable) == {'alu', 'flu_fuu'}
    assert set(summary.particle) == {'pi+', 'pi-', 'both'}
    row = summary.query("observable == 'alu' and target == 'C' and reference == '' "
                        "and view == 'zavg' and particle == 'both'").iloc[0]
    assert row.value == pytest.approx(.34)
    assert row.stat_err == pytest.approx(.02 / np.sqrt(2.5))
    assert row.bin_center == .1
    assert pd.isna(row.z) and pd.isna(row.plot_z)
    assert row.z_selection == '0.5|0.67'
    difference = summary.query("observable == 'alu' and target == 'C' and reference == 'LH2' "
                               "and view == 'zavg' and particle == 'both'").iloc[0]
    assert difference.value == pytest.approx(.2)
    assert difference.stat_err == pytest.approx(.02 / np.sqrt(1.25))
    real = summary.query("observable == 'alu' and target == 'C' and reference == '' "
                         "and view == 'z' and particle == 'both'")
    np.testing.assert_allclose(real.plot_z, [.53, .71])
    scaled = summary.query("observable == 'flu_fuu' and target == 'C' and reference == '' "
                           "and view == 'zavg' and particle == 'both'").iloc[0]
    assert scaled.value == pytest.approx(row.value / np.sqrt(2 * .59 * .41))


def test_summary_respects_single_particle_targets_and_disabled_pages():
    from plot_nuclear_dependence import _plot_summary
    summary = _plot_summary(sample().query("particle == 'pi-'"), ['C'], 'missing', None)
    assert len(summary) == 2
    assert summary.target.eq('C').all()
    assert summary.particle.eq('pi-').all()
    assert summary.observable.eq('alu').all()
    assert summary.view.eq('z').all()
    assert summary.reference.eq('').all()
    np.testing.assert_allclose(summary.plot_z, summary.z)


def test_main_always_writes_summary_beside_pdf_without_systematics(tmp_path, monkeypatch):
    import sys
    import plot_nuclear_dependence as plotting
    out = tmp_path / 'plots' / 'selected.pdf'
    monkeypatch.setattr(plotting, '_load_all', lambda *args: sample())
    monkeypatch.setattr(sys, 'argv', ['plot_nuclear_dependence.py', '--targets', 'C',
                                     '--reference', 'missing', '--particle', 'pi+',
                                     '--x', '.25', '--output', str(out)])
    plotting.main()
    assert out.exists()
    summary = pd.read_csv(out.with_name('selected_summary.csv'))
    assert len(summary) == 2
    np.testing.assert_allclose(summary.value, .3)


def test_z_thpq_parser_canonicalizes_order_and_rejects_bad_pairs():
    from plot_nuclear_dependence import _parse_z_thpq
    assert _parse_z_thpq(['0.5:5.2;-0.8;2.0', '0.67:-0.8;2']) == {
        .5: (-.8, 2., 5.2), .67: (-.8, 2.)}
    for pairs in [['.5'], ['.5:2;2'], ['.5:nan'], ['.5:2', '.5:3']]:
        with pytest.raises(ValueError):
            _parse_z_thpq(pairs)


def write_combined_input(directory, thpq, encoded, charge='pip', suffix=''):
    df = sample().query("target == 'C' and z == .5 and particle == 'pi+'" ).copy()
    df['particle'] = 'pi+' if charge == 'pip' else 'pi-'
    df['process'] = 'sidis'
    df['thpq'] = thpq
    df['variable'] = 'pt'
    path = directory / f'C_{charge}_e10p7_x0p25_q23p3_z0p5{suffix}_thpq{encoded}_binned_summary.csv'
    df.to_csv(path, index=False)
    return path


def test_loader_matches_complete_filename_set_before_reading(tmp_path, monkeypatch):
    import plot_nuclear_dependence as plotting
    write_combined_input(tmp_path, '-0.8|2.0', 'm0p8AND2p0')
    path = write_combined_input(tmp_path, '5.2|-0.8|2.0', '5p2ANDm0p8AND2p0')
    monkeypatch.setattr(plotting, 'COMBINED_DIR', tmp_path)
    df = plotting._load_all('pt', 'sidis', {.5: (-.8, 2., 5.2)}, .25)
    assert len(df) == 1
    assert df.thpq.tolist() == ['-0.8;2.0;5.2']
    assert df._source_file.tolist() == [path.name]
    with pytest.raises(ValueError, match='No combined summary files match'):
        plotting._load_all('pt', 'sidis', {.5: (-.8,)}, .25)


def test_loader_checks_filename_against_csv_metadata(tmp_path, monkeypatch):
    import plot_nuclear_dependence as plotting
    path = write_combined_input(tmp_path, '-0.8|2.0', 'm0p8AND2p0AND5p2')
    monkeypatch.setattr(plotting, 'COMBINED_DIR', tmp_path)
    with pytest.raises(ValueError, match='thpq mismatch'):
        plotting._load_all('pt', 'sidis', {.5: (-.8, 2., 5.2)}, .25)
    df = pd.read_csv(path)
    df['thpq'] = '-0.8|2.0|5.2'
    df['z'] = .67
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match='z mismatch'):
        plotting._load_all('pt', 'sidis', {.5: (-.8, 2., 5.2)}, .25)


def test_selected_inputs_reject_duplicates_mixed_sets_and_missing_charges():
    from plot_nuclear_dependence import _validate_selected_inputs
    df = sample()
    df['thpq'] = '-0.8;2.0'
    with pytest.raises(ValueError, match='Duplicate selected bins'):
        _validate_selected_inputs(pd.concat([df, df.iloc[:1]]), ['C'], 'both', None, 'LH2')
    mixed = df.copy()
    mixed.loc[mixed.particle == 'pi+', 'thpq'] = '-0.8;2.0;5.2'
    with pytest.raises(ValueError, match='Mixed thpq'):
        _validate_selected_inputs(mixed, ['C'], 'both', None, 'LH2')
    with pytest.raises(ValueError, match='No matching z/thpq input'):
        _validate_selected_inputs(df[df.particle == 'pi+'], ['C'], 'both',
                                   {.5: (-.8, 2.)}, 'LH2')


def test_summary_single_thpq_column_tracks_z_average_provenance():
    from plot_nuclear_dependence import _plot_summary
    df = sample()
    df['thpq'] = np.where(df.z == .5, '-0.8;2.0;5.2', '-0.8;2.0')
    summary = _plot_summary(df, ['LH2', 'C'], 'LH2', [.5, .67], .59)
    assert summary.query("view == 'z' and z == .5").thpq.eq('-0.8;2.0;5.2').all()
    assert summary.query("view == 'z' and z == .67").thpq.eq('-0.8;2.0').all()
    assert summary.query("view == 'zavg'").thpq.eq('0.5:-0.8;2.0;5.2|0.67:-0.8;2.0').all()


def test_both_charge_loader_prefers_exact_then_largest_sorted_subset(tmp_path, monkeypatch):
    import plot_nuclear_dependence as plotting
    write_combined_input(tmp_path, '-0.8|2.0|5.2', 'm0p8AND2p0AND5p2', charge='pip')
    write_combined_input(tmp_path, '-0.8|2.0', 'm0p8AND2p0', charge='pim')
    write_combined_input(tmp_path, '-0.8|5.2', 'm0p8AND5p2', charge='pim')
    write_combined_input(tmp_path, '2.0', '2p0', charge='pim')
    write_combined_input(tmp_path, '-0.8|2.0|9.0', 'm0p8AND2p0AND9p0', charge='pim')
    monkeypatch.setattr(plotting, 'COMBINED_DIR', tmp_path)
    requested = {.5: (-.8, 2., 5.2)}
    df = plotting._load_all('pt', 'sidis', requested, .25, allow_subsets=True)
    assert df.query("particle == 'pi+'" ).thpq.tolist() == ['-0.8;2.0;5.2']
    assert df.query("particle == 'pi-'" ).thpq.tolist() == ['-0.8;2.0']
    assert df.query("particle == 'pi-'" ).missing_thpq.tolist() == ['pi-:0.5,5.2']
    plotting._validate_selected_inputs(df, ['C'], 'both', requested, 'missing')
    exact_only = plotting._load_all('pt', 'sidis', requested, .25)
    assert exact_only.particle.tolist() == ['pi+']
    summary = plotting._plot_summary(df, ['C'], 'missing', [.5])
    both = summary.query("particle == 'both'")
    assert both.missing_thpq.eq('pi-:0.5,5.2').all()
    assert both.query("view == 'z'").thpq.tolist() == ['-0.8;2.0;5.2']
    assert both.query("view == 'zavg'").thpq.tolist() == ['0.5:-0.8;2.0;5.2']


def test_summary_reports_reference_and_multiple_z_omissions():
    from plot_nuclear_dependence import _plot_summary
    df = sample()
    df['thpq'] = '-0.8;2.0'
    df['_preferred_thpq'] = '-0.8;2.0;5.2'
    df['missing_thpq'] = [f'{p}:{z:g},5.2' if p == 'pi-' else ''
                          for p, z in zip(df.particle, df.z)]
    summary = _plot_summary(df, ['C'], 'LH2', [.5, .67])
    row = summary.query("reference == 'LH2' and particle == 'both' and view == 'zavg'").iloc[0]
    assert row.missing_thpq == 'pi-:0.5,5.2|pi-:0.67,5.2|LH2/pi-:0.5,5.2|LH2/pi-:0.67,5.2'
    assert row.thpq == '0.5:-0.8;2.0;5.2|0.67:-0.8;2.0;5.2'
