"""Monitor exact-replay UNETR Adam diagnostics in the current session."""
from scripts.mni_constraints import monitor_unetr_combinations as monitor
from scripts.mni_constraints import unetr_optimizer_probe_study as study
from scripts.mni_constraints import analyze_unetr_optimizer_probe as analysis


if __name__=='__main__':
    monitor.NAME=study.NAME;monitor.RESULT_FILENAME='optimizer_probe_results.json';monitor.analysis=analysis
    monitor.compact_result=lambda r:dict(id=r['id'],exact_replay=r['exact_replay'],summary=r['summary'],
                                       virtual_adam_relative_error=r['maximum_virtual_adam_relative_error'])
    monitor.main()
