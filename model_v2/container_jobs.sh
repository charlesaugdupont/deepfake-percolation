#!/bin/bash
# Container share of the Twitter-network experiments (run sequentially; each step logs its own timing).
cd /home/claude/work
CS=0.04329; CLO=0.01233; CHI=0.15199
for c in $CS $CLO $CHI; do
  [ -f out/time_twitter_c$c.pkl ] || python3 exp_time.py twitter $c 0.3 20 out/time_twitter_c$c.pkl
done
[ -f out/mech_twitter.pkl ] || python3 exp_mechanism.py twitter 6 out/mech_twitter.pkl 0.01233,0.0231,0.04329,0.08111,0.15199 0.3
[ -f out/two_twitter.parquet ] || python3 exp_two.py twitter 6 out/two_twitter $CS
[ -f out/priors_twitter.parquet ] || python3 exp_priors.py twitter 2 out/priors_twitter.parquet 0.3 $CS
[ -f out/checkers_twitter.parquet ] || python3 exp_checkers.py twitter 3 out/checkers_twitter 0.3
[ -f out/robust_twitter.parquet ] || python3 exp_robust.py twitter 3 out/robust_twitter.parquet 10000 $CLO,$CS
[ -f out/phase_twitter_rewired.parquet ] || LOSSES=2 python3 exp_phase.py twitter_rewired 3 out/phase_twitter_rewired.parquet
[ -f out/phase_fb_jhu.parquet ] || LOSSES=2 python3 exp_phase.py fb_jhu 4 out/phase_fb_jhu.parquet
echo CONTAINER_ALL_DONE
