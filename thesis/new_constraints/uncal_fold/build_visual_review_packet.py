"""Build a raw-image reviewer page without AI candidates or MSD labels."""
import html
import json

from .prepare_visual_review import OUTPUT


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Uncal fold — raw MRI review</title>
<style>body{font:17px/1.5 system-ui;max-width:1500px;margin:40px auto;padding:0 20px;color:#20252b}
img{width:100%;height:auto}section{border-top:1px solid #ccc;margin-top:45px}a{color:#185ca2}
nav{display:flex;gap:16px;flex-wrap:wrap}</style>
<h1>Uncal fold: raw MRI review</h1>
<p>12 predetermined fold-0 training crops. This page displays raw images only,
without segmentation labels, model predictions, AI candidates, or comparison results.</p>
<p>Native RAS, 1 mm. Coronal panels run anterior to posterior (decreasing y).
Sagittal panels show native y on the horizontal axis. Use full-resolution images
and an MRI viewer for anatomical adjudication; these montages are a review aid.</p>
<p>Record suspected fold visibility and disappearance, candidate last-visible y,
an uncertainty interval, confidence, and reasons for abstention. Do not assume
the double-level appearance persists throughout the entire anterior hippocampus.
The last-visible slice is retained in the anterior partition. Not-resolved is
not evidence of anatomical absence. Use a fresh copy of the blank template.</p>
<p><a href="expert_annotation_template.json">Blank annotation template</a></p><nav>''']
    for name in manifest["cases"]:
        parts.append(f'<a href="#{name}">{html.escape(name)}</a>')
    parts.append('</nav>')
    for name in manifest["cases"]:
        parts.append(f'<section id="{name}"><h2>{html.escape(name)}</h2>')
        for view in ("coronal", "sagittal"):
            filename = f"{name}_{view}.png"
            parts.append(f'<h3>{view.capitalize()}</h3><a href="{filename}"><img loading="lazy" src="{filename}" alt="{name} raw {view} slices"></a>')
        parts.append('</section>')
    parts.append('</html>')
    (OUTPUT / "review_raw.html").write_text('\n'.join(parts))
    template = dict(reviewer_id=None,reviewer_qualification=None,
                    instructions="Complete independently before opening AI annotations or label_comparison.json. Preserve original AI files. Mark unresolved cases as abstentions.",
                    records=[dict(case=name,candidate_last_visible_y=None,candidate_interval_y=None,
                                  abstain=None,confidence=None,slice_observations=[],notes=None)
                             for name in manifest["cases"]])
    template_path = OUTPUT / "expert_annotation_template.json"
    if not template_path.exists():
        template_path.write_text(json.dumps(template,indent=2)+'\n')
    print(OUTPUT / "review_raw.html")


if __name__ == "__main__":
    main()
