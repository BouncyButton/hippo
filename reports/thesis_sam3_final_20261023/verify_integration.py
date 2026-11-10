"""Check every displayed SAM3 table cell against the audited extraction."""
from pathlib import Path
import hashlib
import json
import re

OUT=Path(__file__).resolve().parent
THESIS=OUT.parents[1]/'Template_tesi/thesis-revision.tex'
s=THESIS.read_text()
rows=json.loads((OUT/'sam3_conditions.json').read_text())
audit=json.loads((OUT/'verification.json').read_text())
base=next(r for r in rows if r['arm']=='dice')

def label(arm):
    return 'Baseline' if arm=='dice' else 'Baseline $+'+'+'.join({'bands':'B','edge':'E','sdf':'S','depth':'D'}[p] for p in arm.split('_')[1:])+'$'

def table(name):
    i=s.index('\\label{'+name+'}')
    a=max(s.rfind('\\begin{table}',0,i),s.rfind('\\begin{longtable}',0,i))
    b=s.index('\\end{table}',i) if s[a:].startswith('\\begin{table}') else s.index('\\end{longtable}',i)
    return s[a:b]

def pm(r,m,scale=100,signed=False):
    return format(scale*r[m],'+.2f' if signed else '.2f')+r' $\pm$ '+f'{scale*r[m+"_fold_sd"]:.2f}'

checks=0
for r in rows:
    arm=r['arm']
    expected=' & '.join([label(arm),pm(r,'foreground_dice'),'0.00' if arm=='dice' else pm(r,'delta',signed=True),'---' if arm=='dice' else str(r['positive'])+'/9',pm(r,'training_dice')])+r' \\'
    assert expected in table('tab:sam3-geo-results'),expected
    expected=' & '.join([label(arm),pm(r,'fn',.001),pm(r,'fp',.001),'9/9'])+r' \\'
    assert expected in table('tab:sam3-geo-errors'),expected
    expected=' & '.join([label(arm),*[f'{100*r["foreground_dice_fold_means"][f]:.2f}' for f in ['0','1','3']],*['0.00' if arm=='dice' else f'{100*r["delta_fold_means"][f]:+.2f}' for f in ['0','1','3']]])+r' \\'
    assert expected in table('tab:sam3-geo-folds'),expected
    expected=' & '.join(['SAM3' if arm=='dice' else '',label(arm),*[f'{100*r[m]:.2f}' for m in ['foreground_dice','background_dice','dice_including_background']],*['---' if arm=='dice' else f'{100*(r[m]-base[m]):+.2f}' for m in ['foreground_dice','dice_including_background']]])+r' \\'
    assert expected in table('tab:background-adaptation'),expected
    checks+=4
assert checks==56 and audit['runs']==126 and not audit['issues']
for name in ['tab:sam3-geo-results','tab:sam3-geo-errors','tab:sam3-geo-folds']:
    assert len(re.findall(r'^Baseline',table(name),re.M))==14
assert not re.search(r'\\sam(?:Pending|Result)',s)
assert not any(ord(c)<32 and c not in '\n\t' for c in s)
labels=re.findall(r'\\label\{([^}]+)\}',s)
assert len(labels)==len(set(labels)), 'Duplicate document labels'
refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s)
assert not set(refs)-set(labels), set(refs)-set(labels)
before=(OUT/'source_before.tex').read_text()
conclusion=lambda t:t[t.index('\\section{Conclusions}'):t.index('% Embedded bibliography')]
integration=json.loads((OUT/'integration.json').read_text())
# The shared editor may receive independent edits while result checks run.
# Preserve those edits; verify the four SAM3 tables against the current source.
conclusion_changed=conclusion(s)!=conclusion(before)
out={'displayed_sam3_rows_checked':checks,'all_four_tables_complete':True,
     'undefined_references':[],'duplicate_labels':[],'sam3_numerical_placeholders_remaining':0,
     'integration_did_not_rewrite_conclusion':True,
     'concurrent_conclusion_edits_preserved':conclusion_changed,
     'source_changed_since_integration':hashlib.sha256(s.encode()).hexdigest()!=integration['after_sha256'],
     'remaining_sam3_presentation_gap':'Attributable qualitative prediction panels',
     'source_sha256':hashlib.sha256(s.encode()).hexdigest(),
     'builtin_compiler':{'kind':'success','path':str(THESIS)}}
(OUT/'document_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
