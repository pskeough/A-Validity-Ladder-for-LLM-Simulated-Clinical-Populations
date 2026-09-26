"""Run the whole analysis in dependency order. Local CPU only; no script calls an API.

    python scripts/00_download_nhanes.py
    python scripts/run_all.py            # everything, about 2 hours on 7 cores
    python scripts/run_all.py --fast     # skips 83b and 83e (the controls simulation, about 70 minutes)

83c reads 83b's replicate files, which ship in analysis/brm/83b_reps, so --fast still rebuilds the
controls summary. Each script prints its own receipts and stops on a failed check.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = [
    "76_build_corpus_v3.py",
    # gate
    "82a_gate_gstudy.py", "82b_gate_reml_check.py", "82c_gate_bandflip.py", "82d_gate_decoding_link.py",
    "82e_gate_december.py", "82f_gate_license.py",
    # level 1
    "79a_l1_grm_fit.py", "79b_l1_gateway.py", "79c_l1_personfit.py", "79d_l1_gateway_null.py",
    "79e_l1_unit_controls.py", "79f_l1_summary.py",
    # level 2
    "78a_l2_population_reference.py", "78b_l2_simulated_gaps.py", "78c_l2_r3_verdicts.py",
    "78d_l2_prompt_control.py", "78e_l2_external_r3.py", "78f_l2_report_tables.py",
    # level 3
    "80a_l3_nhanes_frame.py", "80b_l3_sim_cells.py", "80c_l3_estimates.py", "80d_l3_summary.py",
    # level 4
    "81b_l4_structure.py", "81c_l4_invariance.py", "81d_l4_floor_null.py", "81e_l4_summary.py",
    # controls
    "83a_controls_receipt.py", "83b_controls_simulate.py", "83c_controls_summary.py", "83d_l2_se_check.py",
    "83e_l2_power.py", "83f_l2_parametric_power.py", "83g_l2_conditional_verdicts.py",
]
SLOW = {"83b_controls_simulate.py", "83e_l2_power.py"}


def main():
    fast = "--fast" in sys.argv
    env = dict(os.environ, PYTHONUTF8="1")
    for s in ORDER:
        if fast and s in SLOW:
            print(f"skip  {s}")
            continue
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, s)], env=env, capture_output=True, text=True)
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'}  {s}  {time.time() - t:.0f}s", flush=True)
        if r.returncode:
            sys.stdout.write(r.stdout[-3000:])
            sys.stderr.write(r.stderr[-3000:])
            sys.exit(1)


if __name__ == "__main__":
    main()
