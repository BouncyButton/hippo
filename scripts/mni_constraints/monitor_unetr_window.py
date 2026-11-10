"""Monitor the authorized finite-window trials in the active session."""
from scripts.mni_constraints import monitor_unetr_combinations as monitor
from scripts.mni_constraints import unetr_window_study as study
from scripts.mni_constraints import analyze_unetr_window_study as analysis

if __name__=='__main__':
    monitor.NAME=study.NAME;monitor.RESULT_FILENAME='window_results.json';monitor.analysis=analysis
    monitor.main()
