"""Optional laptop archive mirror; never delivers models or submits jobs."""
from scripts.cross_dataset_campaign import collect

original_snapshot = collect.snapshot


def snapshot(stage):
    state = original_snapshot(stage)
    state.pop('base_request.json', None)
    return state


if __name__ == '__main__':
    collect.snapshot = snapshot
    collect.resume_after_archive = lambda *args: None
    collect.main()
