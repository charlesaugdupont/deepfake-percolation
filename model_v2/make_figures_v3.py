"""Render every v3 paper figure (PNG + PDF) from the collected results.
Usage: python3 make_figures_v3.py <data dir, e.g. ../data/v3> <output dir, e.g. ../figures/paper_v3> [--only fig3,figS2]
Figures whose input is missing are skipped with a note."""
import sys, os, argparse, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_save = mfig.Figure.savefig


def _save_both(self, fname, *a, **k):
    _save(self, fname, *a, **k)
    if str(fname).endswith(".png"):
        _save(self, str(fname)[:-4] + ".pdf", *a, **k)


mfig.Figure.savefig = _save_both


def figures(d, o):
    j = lambda *p: os.path.join(d, *p)
    t = lambda name: os.path.join(o, name)
    import v3_fig1_mechanism as f1, v3_fig2_time as f2, v3_fig_two_time as f3, v3_fig3_regimes as f4
    import v3_figS_starting as s_start, v3_figS_checkers as s_chk, v3_figS_stakes as s_stk, v3_figS_robust as s_rob
    import v3_figS_intervention as s_cut, v3_figS_settings as s_set, v3_fig6_two as s_two, v3_figS_networks as s_net
    import v3_figS_gsa as s_gsa, v3_figS_inequality as s_ineq, v3_figS_gss as s_gss
    gss = os.path.join(d, "..", "gss", "gss_press_extract.parquet")       # survey extract (gss_press.py), beside data/v3
    return [
        # main text (4 figures)
        ("fig1", [j("mech_twitter_agents.pkl")], lambda: f1.make(j("mech_twitter_agents.pkl"), t("fig1_model_mechanism.png"))),
        ("fig2", [j("time_twitter_agents.pkl")],                      # the 4,000-item runs (time_long) when present (6 Nov)
         lambda: f2.make(j("time_long_twitter_agents.pkl") if os.path.exists(j("time_long_twitter_agents.pkl"))
                         else j("time_twitter_agents.pkl"), t("fig2_one_checker_over_time.png"))),
        ("fig3", [j("two_time_twitter_agents.pkl"), j("two_twitter.parquet")], lambda: f3.make(d, t("fig3_two_checkers.png"))),
        ("fig4", [j("phase_twitter.parquet")],
         lambda: f4.make_main(j("phase_twitter.parquet"), t("fig4_regimes_and_consequences.png"), "twitter",
                              cache=j("theory_twitter_all.json"))),
        # SI, grouped by the main figure they support. Model (Fig. 1): S1-S3 and Table S1; one checker (Fig. 2): S4-S5;
        # two checkers (Fig. 3): S6-S7; regimes and consequences (Fig. 4): S8-S10. The cost-cut figure (former S5) was
        # dropped on 5 Nov (outcomes barely change; v3_figS_intervention.py still draws it) and S6-S11 renumbered S5-S10.
        ("figS1", [j("priors_twitter.parquet")],
         lambda: s_start.make(j("priors_twitter.parquet"), t("figS1_starting_beliefs.png"), j("priors_hex_twitter.parquet"))),
        ("figS2", [j("checkers_twitter.parquet")], lambda: s_chk.make(j("checkers_twitter"), t("figS2_checker_quality.png"))),
        ("figS3", [j("penalty_twitter_agents.pkl"), j("stakes_twitter_agents.pkl"), j("penalty_two_twitter_agents.pkl"),
                   j("stakes_two_twitter_agents.pkl")],
         lambda: s_stk.make(d, t("figS3_stakes.png"), t("tableS1_stakes"))),
        ("figS4", [j("robust_twitter.parquet")], lambda: s_rob.make(j("robust_twitter.parquet"), t("figS4_robustness.png"), 0.05)),
        ("figS5", [j("time_twitter_phi06_agents.pkl"), j("priors_settings_twitter.parquet")],
         lambda: s_set.make_one(d, t("figS5_other_settings_one_checker.png"))),
        ("figS6", [j("two_twitter.parquet"), j("two_hk.parquet"), j("two_fb_jhu.parquet"), j("two_twitch.parquet")],
         lambda: s_two.make_networks(d, t("figS6_two_checkers_networks.png"))),
        ("figS7", [j("two_time_twitter_agents.pkl"), j("two_asym_settings_twitter_agents.pkl")],
         lambda: s_set.make_two(d, t("figS7_other_settings_two_checkers.png"))),
        ("figS8", [j("phase_twitter.parquet"), j("phase_hk.parquet")], lambda: s_net.make(d, t("figS8_networks.png"))),
        ("figS9", [j("phase_twitter.parquet")],
         lambda: f4.make_ratios(j("phase_twitter.parquet"), t("figS9_other_loss_gain_ratios.png"), net="twitter",
                                cache=j("theory_twitter_all.json"))),
        ("figS10", [j("gsa_twitter.parquet")], lambda: s_gsa.make(j("gsa_twitter.parquet"), t("figS10_sensitivity.png"))),
        # added 6 Nov (provisional number): audience inequality vs average audience on random networks
        ("figS11", [j("ineq_cv.parquet"), j("ineq_k.parquet"), j("synthetic_networks.csv"), j("real_network_degrees.csv")],
         lambda: s_ineq.make(d, t("figS11_inequality.png"))),
        # added 6 Nov (provisional number): survey data and model side by side (GSS confidence in the press by party;
        # model trust in the checker by audience size, from the Fig. 2 runs)
        ("figS12", [j("time_deg_twitter_agents.pkl"), gss], lambda: s_gss.make(d, t("figS12_survey_and_model.png"))),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(x for x in a.only.split(",") if x)
    for name, needs, fn in figures(a.data, a.out):
        if only and name not in only:
            continue
        missing = [p for p in needs if not os.path.exists(p)]
        if missing:
            print(f"{name}: skipped (missing {', '.join(os.path.basename(m) for m in missing)})"); continue
        try:
            fn()
        except Exception:
            print(f"{name}: FAILED"); traceback.print_exc()
