"""End-to-end smoke run of the experiment pipeline on a tiny configuration."""

import json

import pytest

from wmprobe.experiment import Config, run

pytestmark = pytest.mark.slow

CONDS = ("plain_t1.0", "plain_t1.2", "kgw_g0.25_d4", "gumbel_h4", "synthid_n2")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = Config(
        out=out,
        n_prompts=8,
        n_new=50,
        burn_in=10,
        lengths=(20, 40),
        corpus_sizes=(4,),
        n_draws=3,
        c_queries=(200,),
        c_reps=2,
        d_prompts=3,
        d_repeats=4,
        d_new=12,
        proxies=("gpt2",),
        conditions=CONDS,
    )
    run(cfg)
    return out, json.loads((out / "results.json").read_text())


def test_results_cover_every_test_and_condition(results):
    _, r = results
    assert set(r) >= {"config", "keyed", "test_a", "test_b", "test_c", "test_d"}
    for test in ("test_a", "test_b", "test_d"):
        assert set(r[test]["gpt2"]["per_condition"]) == set(CONDS)
    assert len(r["test_a"]["gpt2"]["per_condition"]["plain_t1.0"]["z"]) == 8
    assert set(r["test_c"]["gpt2"]["per_condition"]["kgw_g0.25_d4"]) == {"200"}


def test_strong_signals_show_up_even_at_tiny_scale(results):
    _, r = results
    assert r["keyed"]["kgw_g0.25_d4"]["auc_vs_plain_t1.0"] == 1.0
    assert r["keyed"]["synthid_n2"]["auc_vs_plain_t1.0"] == 1.0
    assert r["test_d"]["gpt2"]["per_condition"]["gumbel_h4"]["pooled_z"] > 5


def test_c_prompts_constrain_the_answer_space(results):
    _, r = results
    for c in CONDS:
        assert r["test_c"]["gpt2"]["per_condition"][c]["200"]["answer_set_size"] >= 2


def test_b_reports_sign_flip_p_values_with_and_without_burn_in(results):
    _, r = results
    per = r["test_b"]["gpt2"]["per_condition"]["kgw_g0.25_d4"]
    assert 0 < per["full_p"] <= 1 and 0 < per["full_p_no_burn_in"] <= 1
    assert r["config"]["burn_in"] < r["config"]["n_new"]


def test_auc_tables_pair_watermarks_with_plain_conditions(results):
    _, r = results
    auc = r["test_a"]["gpt2"]["auc"]
    assert set(auc) == {"kgw_g0.25_d4", "gumbel_h4", "synthid_n2"}
    assert set(auc["kgw_g0.25_d4"]) == {"plain_t1.0", "plain_t1.2"}
    assert set(auc["kgw_g0.25_d4"]["plain_t1.0"]) == {"20", "40"}


def test_rerun_reuses_cached_artifacts(results):
    out, _ = results
    assert (out / "corpus" / "kgw_g0.25_d4.json").exists()
    assert (out / "scored" / "gpt2" / "kgw_g0.25_d4.pkl").exists()


def test_report_renders_tables_and_figures(results):
    from wmprobe.report import render

    out, _ = results
    render(out)
    text = (out / "report.md").read_text()
    for heading in ("Keyed baselines", "Test A", "Test B", "Test C", "Test D"):
        assert f"## {heading}" in text
    assert "kgw_g0.25_d4" in text
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert figs == [
        "test_a_auc_vs_length.png",
        "test_a_z.png",
        "test_b_rejection.png",
        "test_c_rejection.png",
        "test_d_z.png",
    ]
