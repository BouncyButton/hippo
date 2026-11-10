"""Submit once after the conservative capacity check; no local collector."""
import json
from scripts.geofwi_constraints.prepare import ROOT, NAME, REPORT, REMOTE
from scripts.geofwi_constraints.transfer import remote
from scripts.geofwi_cluster.campaign import read, write


def main():
    p=read(REPORT/"preparation.json")
    assert read(REPORT/"transfer_receipt.json")["manifest_sha256"]==p["manifest_sha256"]
    code="""import json,sys,time
from pathlib import Path
stage=Path(sys.argv[1]);sys.path.insert(0,str(stage))
import campaign
assert campaign.sha(stage/'manifest.json')==sys.argv[2]
usage=campaign.runtime.quota()
# Frozen source is already staged; remaining overhead is two active runs and
# their temporary recovery copies, plus the completed small result records.
assert usage['free_gib']>=54,usage
audit=dict(status='passed',quota=usage,active_runs_logical_bound_gib=22,
           mirrored_active_bound_gib=44,all_metadata_allowance_gib=4,
           initial_total_peak_additional_bound_gib=48,
           remaining_peak_additional_after_staging_bound_gib=46,
           remaining_headroom_at_peak_gib=usage['free_gib']-46,
           basis='Measured largest latest checkpoint 1.383 GiB; six simultaneous latest-sized copies plus two best-sized copies per lane, rounded up to 11 GiB/lane; two lanes; factor-two storage allowance.',
           supersedes='Earlier 14 GiB logical-active estimate in frozen planning records',
           user_storage_policy='Cluster archives; no laptop collector or acknowledgment',time=time.time())
campaign.write(stage/'capacity_audit.json',audit)
result=campaign.submit(stage)
print(json.dumps(dict(submission=result,capacity_audit=audit)))
"""
    # Any unknown outcome must be reconciled using remote submissions/initial.
    result=json.loads(remote(code,REMOTE,p["manifest_sha256"]))
    write(REPORT/"submission.json",result["submission"])
    write(ROOT/"experiments"/NAME/"submission.json",result["submission"])
    write(REPORT/"capacity_audit.json",result["capacity_audit"])
    print(json.dumps(result))


if __name__=="__main__":
    main()
