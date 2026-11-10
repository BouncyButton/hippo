"""Monitor original subfield-sensitive losses through sequential confirmation."""
from scripts.mni_constraints import monitor_unetr_combinations as monitor
from scripts.mni_constraints import unetr_subfield_constraints_study as study
from scripts.mni_constraints import analyze_unetr_subfield_constraints as analysis

if __name__=='__main__':
    monitor.NAME=study.NAME;monitor.RESULT_FILENAME='subfield_results.json';monitor.analysis=analysis
    monitor.main()
