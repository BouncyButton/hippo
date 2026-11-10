"""Monitor readiness trials in the current session through verified completion."""
from scripts.mni_constraints import monitor_unetr_combinations as monitor
from scripts.mni_constraints import unetr_readiness_study as study
from scripts.mni_constraints import analyze_unetr_readiness_study as analysis


if __name__=='__main__':
    monitor.NAME=study.NAME
    monitor.RESULT_FILENAME='readiness_results.json'
    monitor.analysis=analysis
    monitor.main()
