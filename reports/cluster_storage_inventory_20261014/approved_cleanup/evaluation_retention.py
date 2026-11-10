"""Archive, verify and retire only the generated evaluation supplement."""
import importlib.util,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
mode=sys.argv[1]
sys.argv=[str(BASE/'archive_cleanup.py'),mode+'_results']
spec=importlib.util.spec_from_file_location('archive_worker',BASE/'archive_cleanup.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.BASE=BASE/'evaluation_retention';a.BASE.mkdir(exist_ok=True)
a.TARGETS=['storage_retirement_audit_20261014']
a.GROUPS={'evaluation':a.TARGETS}
{'inventory':a.inventory,'archive':a.archive,'verify':a.verify,'delete':a.delete}[mode]()
