"""Monitor fresh controls and paired simplification results until resolved."""
from scripts.mni_constraints import monitor_unetr_combinations as monitor
from scripts.mni_constraints import unetr_plain_diceonly_study as study
from scripts.mni_constraints import analyze_unetr_plain_diceonly_study as analysis

if __name__ == '__main__':
    monitor.NAME=study.NAME
    monitor.RESULT_FILENAME='plain_diceonly_results.json'
    monitor.analysis=analysis
    monitor.main()
