import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_degree_edge_lowdata import weighted_config
from edge_lowdata_metrics import compare_histories
from audit_edge_generalization import compare_splits, METRICS


def test_only_declared_weighting_changes_run_and_scalar_coefficients_survive():
    reference={'run':{'constraint_set':'bands_edge','constraint_config':{'bands_weight':.002},
        'edge_weight':.009,'epochs':75,'training_augmentation':'none','seed':2,
        'early_stopping_patience':8,'early_stopping_min_epochs':60}}
    before=copy.deepcopy(reference)
    actual=weighted_config(reference)
    expected=copy.deepcopy(before['run'])
    expected['constraint_set']='degree_bands_edge_A'
    expected['constraint_config'].update(bands_degree_alpha=2.,bands_degree_normalization='inner')
    assert actual['run']==expected and reference==before


def test_weighted_candidate_compares_against_edge_not_itself():
    histories={k:[{'epoch':i+1,'val_dice_hard':v} for i,v in enumerate(values)]
        for k,values in [('edge',[.5,.6,.7]),('weighted',[.6,.7,.8])]}
    report=compare_histories(histories,candidate='weighted',references=('edge',))
    assert report['thresholds']['edge_reference_best']==.7
    assert report['models']['weighted']['crossings']['edge_reference_best']['first']['first_epoch']==2
    assert abs(report['weighted_minus_reference']['edge']['best_dice']-.1)<1e-10
    def records(prefix):
        return {k:[{'case_name':prefix+str(i),'metrics':{m:float(row['val_dice_hard']) for m in METRICS}}
                   for i,row in enumerate(rows)] for k,rows in histories.items()}
    gaps=compare_splits(records('train'),records('val'),candidate='weighted',references=('edge',))
    assert abs(gaps['weighted_effect_relative_to']['edge']['macro_dice']['train_weighted_minus_reference']-.1)<1e-10
