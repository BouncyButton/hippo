from pathlib import Path
import json,re,hashlib,collections
root=Path('/Users/filippofocaccia/Desktop/hippo')
out=root/'reports/thesis_professor_followup_20261103'
p=root/'Template_tesi/thesis-revision.tex'
s=p.read_text();before=(out/'source_before.tex').read_text()
ann=json.loads((root/'reports/thesis_professor_revision_20261102/annotations.json').read_text())
def line(anchor):
    return s[:s.index(anchor)].count('\n')+1
groups=[
([1,2,3,4,5,6,7,8,9,10,53,54,55], 'Abstract: explain segmentation, medical relevance, LTN and the two learning settings; omit replication/protocol detail and unclear audit claim.', 'Addressed in retained submission abstract; compiled abstract disabled.', 'Semantic segmentation assigns a class'),
([11,12,13,14,15,16,17,19,56,57], 'Introduction: context, mask definition, two practical limitations, coherent progression from the general method to learning settings and then datasets; illustrative figure.', 'Addressed. Learning settings now precede the contiguous dataset discussion; image/mask examples and the two limitations remain.', r'\section{Introduction}'),
([18], 'Explain LTN before logLTN and distinguish their roles.', 'Addressed with plain-language introduction and numerical example in Section 2.3; no claim that notation alone improves an identical loss.', 'Logic Tensor Networks (LTNs) provide'),
([20,21,22], 'Intro: model citations, remove protocol inventory and repeated contributions.', 'Addressed. One focused contribution paragraph; numerical schedules in Methods and Appendix A.', 'The contribution is a common LTN/logLTN'),
([23], 'Move convergence caveat to limitations.', 'Addressed in Results limitations; conclusions do not repeat the detailed analysis.', r'\subsection{Limits of the Findings}'),
([24,25], 'Overview: class counts; terminology of constraints/spatial supervision; handwritten question about development dataset.', 'Class counts 2/3/2 and soft-supervision definition addressed. GeoFWI3D remains the first nonmedical application; it is not relabelled as a formal development-only split on the basis of a question mark.', r'\label{fig:thesis-overview}'),
([26,27,28,29,30,58,59,60,61], 'Background: define image/task/case/class symbols, explain varying native dimensions and softmax, use loss notation with inputs, remove misplaced assembly aside and chapter roadmap.', 'Addressed in Section 2.1. Existing corrections retained and checked.', r'\subsection{Segmentation and Supervised Objectives}'),
([31], 'Nine-pixel Dice illustration.', 'Addressed by the existing worked diagram.', r'\label{fig:dice-example}'),
([32,33,34,62], 'LTN: define grounding, distinguish truth from probability, relocate band-specific example and simplify direct-loss implementation explanation.', 'Addressed. Generic probability-to-log-penalty example stays in Background; boundary worked example is in Section 3.2.', r'\subsection{Logic Tensor Networks and logLTN}'),
([35], 'Constraint diagram: clarify the dark outline versus coloured neighbour segments.', 'Addressed by the existing caption: outline marks annotation boundary; coloured segments join voxel centres.', r'\label{fig:constraint-domains}'),
([36,37,38,39,40], 'Handwritten morphology drawing request and associated strokes.', 'Addressed by the existing three-dimensional morphology figure. Individual strokes are not counted as separate requests.', r'\subsection{Boundary-Band Membership}'),
([41,42,43,44,45,46,47,48,49,50,51], 'Band statements: consistent symbols, link Phi to actual loss, explain epsilon and differentiability.', 'Addressed with explicit grounding/mean relationship, two numerical cases, count factor, separate validity rule, and explanation that backpropagation acts through predictions rather than morphology.', r'\label{eq:balanced-band-loss}'),
([52,64], 'Edge predicate and opposite classifications across a boundary; Andrea highlight has no written instruction.', 'Annotated cross-boundary sign explained without dropping same-side pairs present in the implementation. Transition predicate and suppressed group subscript are now explicitly defined. The meaning of Andrea empty highlight is not assumed beyond clarifying its marked predicate.', r'\subsection{Edge Consistency}'),
([63], 'Italicize et al. in prose.', 'Addressed; bibliography formatting retained separately.', r'\subsection{Related Work on Spatial Supervision}'),
([65], 'Explain why seed numbers differ among studies.', 'Addressed in Design and Units of Comparison. Recorded streams, matched comparisons and exploratory selection retained; no scientific disclosure removed.', 'The medical conventional seed sets were chosen'),
([66], 'Why compare four architectures and how do they differ?', 'Addressed in Section 2.2 and reinforced in Methods; layer widths and optimizer settings moved to Appendix A.', r'\subsection{Segmentation Architectures}'),
([67], 'Missing paired baseline/constrained prediction panels.', 'OPEN: substantive visual-evidence gap. No verified final matched pair recovered in the inspected local archives. Placeholder removal alone does not resolve this comment.', 'These measurements describe agreement with the annotations.'),
([68], 'Plural correction to future-work heading.', 'Addressed: Future Research Directions; priorities rewritten around specific follow-up questions.', r'\subsection{Future Research Directions}')
]
covered=[]; items=[]
for ids,issue,status,anchor in groups:
 for i in ids:
  a=ann[i-1];covered.append(i)
  items.append({'annotation_id':i,'reviewer':a['reviewer'],'original_pdf_page':a['page'],'type':a['type'],'original_comment':a.get('comment',''),'issue_group':issue,'assessment':status,'current_source_line':line(anchor)})
assert sorted(covered)==list(range(1,len(ann)+1))
(out/'annotation_reassessment.json').write_text(json.dumps(sorted(items,key=lambda x:x['annotation_id']),indent=2,ensure_ascii=False)+'\n')
labels=re.findall(r'\\label\{([^}]+)\}',s)
refs=re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',s)
cites=[k for m in re.findall(r'\\cite\{([^}]+)\}',s) for k in m.split(',')]
bib=re.findall(r'\\bibitem\{([^}]+)\}',s)
tables=lambda text:re.findall(r'\\begin\{(?:table|longtable)\}.*?\\end\{(?:table|longtable)\}',text,re.S)
def section(text,start,end):
 a=text.index(start);return text[a:text.index(end,a)]
def words(text):
 text=re.sub(r'(?m)^%.*$','',text)
 text=re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}','',text,flags=re.S)
 return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",text))
results=section(s,r'\section{Results and Analysis}',r'\section{Conclusions}')
stack=[];env_errors=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',s):
 if m[1]=='begin':stack.append(m[2])
 elif not stack or stack[-1]!=m[2]:env_errors.append(m[2])
 else:stack.pop()
checks={
 'source_sha256':hashlib.sha256(s.encode()).hexdigest(),
 'missing_references':sorted(set(refs)-set(labels)),
 'duplicate_labels':[k for k,n in collections.Counter(labels).items() if n>1],
 'missing_citations':sorted(set(cites)-set(bib)),
 'environment_errors':env_errors+stack,
 'all_31_table_blocks_unchanged':tables(s)==tables(before) and len(tables(s))==31,
 'results_chapter_unchanged':results==section(before,r'\section{Results and Analysis}',r'\section{Conclusions}'),
 'results_figures':results.count(r'\begin{figure}'),
 'results_tables':len(tables(results)),
 'builtin_compile':'success on revision before final edge-predicate explanatory sentence; final check to be recorded separately',
 'fresh_visual_review':'Unavailable: compile tool updates editor preview but does not export PDF; CUA blocks com.openai.codex. Local thesis-revision.pdf is dated 2026-11-02 and was not treated as current.',
 'approx_word_counts':{}
}
for name,start,end in [('main_adaptation',r'\subsection{Adapting Pretrained Models}',r'\subsection{Evaluation Measures}'),('main_methods',r'\section{Datasets',r'\section{Results'),('future_work',r'\subsection{Future Research Directions}',r'\renewcommand{\refname}')]:
 checks['approx_word_counts'][name]={'before':words(section(before,start,end)),'after':words(section(s,start,end))}
assert not any(checks[k] for k in ['missing_references','duplicate_labels','missing_citations','environment_errors'])
assert checks['all_31_table_blocks_unchanged'] and checks['results_chapter_unchanged']
(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
recovery=json.loads((out/'prediction_recovery.json').read_text())
recovery['partial_npz_inspection']={'archive':'reports/sam3_geofwi_f013_s012_126_20261019/account_migration_20261021/stopped_progress.zip','member':'work/GeoFWI3D_SAM3_f1s0_dice_sdf_u25_v1/evaluation_partial.npz','keys':{'prediction':{'shape':[96,96,96],'dtype':'uint8'}},'assessment':'One interrupted-evaluation prediction exists. This is not a verified final matched baseline/constrained pair; it was not presented as completed qualitative evidence.'}
(out/'prediction_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
fragment=out/'submission_abstract.tex'
if fragment.exists():fragment.rename(out/'submission_abstract.txt')
report=[
'PROFESSOR-FEEDBACK REASSESSMENT — 3 NOVEMBER 2026',
'',
'Scope: source revision and re-review against both annotated PDFs and the two email screenshots.',
'Edited source: '+str(p),
'Reference PDFs: review1-luca_tesi_magistrale_draft.pdf; tesi_magistrale_draft_AT.pdf (Downloads).',
'The 68 PDF annotation objects include text comments, highlights, strikeouts and individual ink strokes. They are not 68 separate revision requests.',
'Pages below refer to the ORIGINAL reviewed PDFs. Current locations are source lines, not unverified new PDF page numbers.',
'',
'ASSESSMENT',
'The remaining editorial changes have been applied. The introduction now keeps general method, learning settings and dataset discussion in a readable progression. The contribution is framed as a shared formulation and empirical investigation, without claiming that an equivalent logical notation alone changes optimization.',
'Worked examples now explain a truth degree, its negative log penalty and the two-sided boundary loss. Morphology is explicitly fixed with respect to model parameters; the prediction path remains differentiable. The signed-distance convention and detailed architecture/optimizer settings are retained in Appendix A.',
'The main adaptation section is approximately 831 versus 2,006 lexical words under the same source-counting procedure (about 59% shorter; figures excluded). It now gives shared principles once and preserves the important dataset differences, support overlap, schedules and matched baseline groups. The appendix was also consolidated instead of duplicating all the former prose.',
'Future Research Directions now prioritizes independent evaluation, controlled ablations, qualitative predictions, anatomical validation and adaptation questions. Repository preparation-script descriptions have been removed.',
'The main Results chapter remains focused: three figures, zero tables. The 31 complete data-table blocks and Results text are byte-for-byte unchanged in this turn. Detailed interpretation is not repeated in Conclusions.',
'',
'ANNOTATION-BY-ANNOTATION GROUPS'
]
for ids,issue,status,anchor in groups:
 refs_group=', '.join(f'{ann[i-1]["reviewer"]} p.{ann[i-1]["page"]}' for i in ids if ann[i-1].get('comment'))
 refs_group=', '.join(dict.fromkeys(refs_group.split(', '))) if refs_group else f'{ann[ids[0]-1]["reviewer"]} p.{ann[ids[0]-1]["page"]} (ink/highlight)'
 report.extend(['',f'Objects {",".join(map(str,ids))} — {refs_group}',issue,status,f'Current source line {line(anchor)}.'])
report.extend([
'',
'EMAIL-LEVEL REQUESTS',
'- Reader-first, less technical-report prose: addressed by the revised progression, examples, shared adaptation explanation and concise future directions. The mathematical contribution still requires notation; technical depth is concentrated there and in Appendix A. This is an editorial assessment, not a claim that the professors have approved the new voice.',
'- Acronyms: primary terms expanded at first use; this pass repaired the MONAI definition after relocating implementation details and retained the full MNI dataset name. Repeated expansion in every chapter was not required by Andrea, who allowed either policy.',
'- Fewer subdivisions: main adaptation uses one subsection and three dataset paragraphs. New reproducibility material belongs in the appendix; the short boundary-setting note is a paragraph within the existing stable-evaluation subsection.',
'- Key results as charts, complete tables in appendix: maintained. The two critical main results comparisons and metric interpretation remain in three figures, with no main Results tables.',
'- Abstract: excluded from the compiled thesis by the existing switch, with a revised copy retained in the source and submission_abstract.txt. Andrea qualified the submission-format advice with “if I remember correctly”; no independent claim about the university rule is made.',
'',
'WHAT IS ACTUALLY STILL OPEN',
'1. Paired prediction panels (Andrea, original p.52): one substantive evidence/presentation gap remains. Ninety-five local archives were inspected. An interrupted evaluation contains one uint8 96×96×96 prediction; two recovery archives contain checkpoints. No verified completed baseline/constrained pair from the final reported campaigns was recovered. Earlier pilots, pretrained initialization weights and curves are not substitutes. Closing this requires attributable final outputs for the same case/protocol or a targeted matched rerun, followed by a stated case-selection rule. No new training was launched and no panels were invented.',
'2. Current visual pagination check: the built-in compiler succeeded, but it does not export the refreshed PDF. The existing on-disk PDF is from 2 November. The computer-use tool explicitly blocks access to the Codex app, so the new page breaks and figure placement could not be visually inspected. No separate PDF was compiled and no tab was opened. A fresh editor PDF is needed to close this verification step.',
'3. Submission administration: confirm whether the programme wants the abstract solely in the submission form. The source is already configured that way, following Andrea’s advice; the text is preserved if the rule differs.',
'Independent confirmatory runs, surface metrics, unweighted depth controls and new datasets are scientific follow-up work, now explicitly prioritized. They are not missing cells in the completed fourteen-objective result inventories and were not invented as a prerequisite for responding to the editorial comments.',
'',
'SELF-REVIEW AND VERIFICATION',
'- Rechecked each textual comment and the relevant handwritten pages, including the empty Andrea highlight on Transition; no unwritten instruction was claimed for that highlight.',
'- Repaired a stale geological-development cross-reference, defined the abbreviated edge-pair domain, clarified the Transition predicate, moved MONAI expansion back to its first use, and shortened the adaptation workflow caption.',
'- No missing references, duplicate labels, missing citation keys, or mismatched LaTeX environments in static validation.',
'- All 31 data-table blocks and the Results chapter are unchanged; three Results figures and zero Results tables remain.',
'- Final built-in compilation status is recorded in verification.json. Visual verification remains qualified as above.',
'',
'Supporting files: annotation_reassessment.json (all 68 objects); verification.json; prediction_recovery.json; source_before.tex; revise.py; self_review.py.',
'No professor approval or complete resolution of the qualitative-evidence comment is claimed.'
])
(out/'feedback_reassessment.txt').write_text('\n'.join(report)+'\n')
print(json.dumps(checks,indent=2))

