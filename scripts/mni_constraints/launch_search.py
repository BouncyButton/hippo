"""Launch the revised target search, using the safely staged recovery artifacts."""
from scripts.mni_constraints import launch_strong
from scripts.mni_constraints.target_search import NAME,MODULE,RECOVERY,verify


if __name__=='__main__':
    launch_strong.NAME=NAME;launch_strong.MODULE=MODULE;launch_strong.verify=verify
    launch_strong.COLLECTOR_MODULE='scripts.mni_constraints.collect_search'
    launch_strong.main()
