"""Reuse verified, previously rendered figures without recompiling a second PDF."""
from pathlib import Path
import hashlib
import json
import re
import sys

sys.path.insert(0, '/private/tmp/thesis_fast_compile_deps')
import fitz
from PIL import Image, ImageDraw

ROOT = Path('/Users/filippofocaccia/Desktop/hippo')
PROJECT = ROOT / 'Template_tesi'
AUDIT = Path(__file__).resolve().parent
SOURCE = PROJECT / 'thesis-revision.tex'
REFERENCE = ROOT / 'reports/thesis_professor_followup_20261103/source_before.tex'
REFERENCE_PDF = PROJECT / 'thesis-revision.pdf'
FIGURES = PROJECT / 'figures/compiled'
EDITABLE = PROJECT / 'figures/source'
PATTERN = re.compile(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', re.S)
# PDF pages are from the hash-verified 2 November source/render pair.
# Each band excludes surrounding prose, caption and page number.
REGIONS = {
    1: (6, 194, 512), 2: (7, 220, 532), 3: (9, 350, 493),
    4: (11, 355, 608), 5: (16, 165, 533), 6: (17, 451, 652),
    7: (31, 299, 607), 9: (43, 168, 557),
    10: (44, 196, 380), 11: (45, 345, 558),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

source = SOURCE.read_text()
reference = REFERENCE.read_text()
validation = json.loads((ROOT / 'reports/thesis_results_readability_20261102/final_validation.json').read_text())
assert sha(REFERENCE.read_bytes()) == validation['source_sha256']
assert REFERENCE_PDF.stat().st_size == 756964
original_blocks = list(PATTERN.finditer(reference))
blocks = list(PATTERN.finditer(source))
assert len(blocks) == 11 and len(original_blocks) == 11
assert not (AUDIT / 'source_before.tex').exists(), 'Do not overwrite the original audit snapshot.'
(AUDIT / 'source_before.tex').write_text(source)
FIGURES.mkdir(parents=True, exist_ok=True)
EDITABLE.mkdir(parents=True, exist_ok=True)
(AUDIT / 'rendered').mkdir(exist_ok=True)
pdf = fitz.open(REFERENCE_PDF)
records = []
replacements = []
previews = []
for number, match in enumerate(blocks, 1):
    if number not in REGIONS:
        continue  # Figure 8 has an updated label and stays live TikZ.
    assert match.group() == original_blocks[number - 1].group(), number
    label = re.search(r'\\label\{fig:([^}]+)\}', source[match.end():]).group(1)
    page_number, top, bottom = REGIONS[number]
    page = pdf[page_number - 1]
    band = fitz.Rect(65, top, 535, bottom)
    boxes = []
    for kind, bounds in page.get_bboxlog():
        rect = fitz.Rect(bounds)
        if rect.intersects(band):
            assert band.contains(rect), (number, kind, rect)
            boxes.append(rect)
    assert boxes
    clip = fitz.Rect(min(r.x0 for r in boxes) - 1.5, min(r.y0 for r in boxes) - 1.5,
                     max(r.x1 for r in boxes) + 1.5, max(r.y1 for r in boxes) + 1.5)
    output = FIGURES / (label + '.pdf')
    figure = fitz.open()
    target = figure.new_page(width=clip.width, height=clip.height)
    target.show_pdf_page(target.rect, pdf, page_number - 1, clip=clip)
    figure.save(output, garbage=4, deflate=True)
    figure.close()
    saved = fitz.open(output)
    assert len(saved) == 1
    text = saved[0].get_text(clip=saved[0].rect)
    assert 'Figure ' not in text, (label, 'Caption was included in the graphic')
    pix = saved[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    preview = AUDIT / 'rendered' / (label + '.png')
    pix.save(preview)
    image = Image.open(preview).convert('RGB')
    image.thumbnail((650, 560))
    tile = Image.new('RGB', (680, 600), 'white')
    ImageDraw.Draw(tile).text((12, 9), f'{number}. {label}', fill='black')
    tile.paste(image, ((680 - image.width) // 2, 32))
    previews.append(tile)
    # Source raster/image objects and vector paths are retained in PDF form.
    assert len(saved[0].get_drawings()) == len(page.get_drawings())
    saved.close()
    (EDITABLE / (label + '.tikz')).write_text(match.group() + '\n')
    rel = output.relative_to(PROJECT).as_posix()
    replacements.append((match.start(), match.end(),
                         '% Cached vector figure; editable source: figures/source/' + label + '.tikz\n'
                         + '\\includegraphics{' + rel + '}'))
    records.append(dict(number=number, label=label, asset=rel,
                        source_sha256=sha(match.group().encode()),
                        asset_sha256=sha(output.read_bytes()), source_pdf_page=page_number,
                        clip_bp=list(clip), width_bp=clip.width, height_bp=clip.height,
                        bytes=output.stat().st_size))

for start, end, replacement in reversed(replacements):
    source = source[:start] + replacement + source[end:]
source = source.replace('\\usepackage{tikz}', '\\usepackage{graphicx}\n\\usepackage{tikz} % Only the compact, editable adaptation workflow is drawn live.', 1)
start = source.index('% Fast, self-contained raster panels')
end = source.index('\\definecolor{thesisBlue}', start)
assert '\\thesisRaster{' not in source[end:]
source = source[:start] + '% Heavy figures are stored in figures/compiled; upload this folder with the TeX source.\n' + source[end:]
source = source.replace('% Standalone bibliography is embedded; the Overleaf template may use ref.bib.',
                        '% Bibliography is embedded; figure assets are in figures/compiled.\n'
                        '% For Overleaf, upload thesis-revision.tex AND figures/compiled together.')
SOURCE.write_text(source)
contact = Image.new('RGB', (2040, 600 * ((len(previews) + 2) // 3)), '#eeeeee')
for i, tile in enumerate(previews):
    contact.paste(tile, ((i % 3) * 680, (i // 3) * 600))
contact.save(AUDIT / 'rendered/contact-sheet.png')
manifest = dict(reference_source_sha256=sha(REFERENCE.read_bytes()),
                reference_pdf_sha256=sha(REFERENCE_PDF.read_bytes()),
                source_before_sha256=sha((AUDIT / 'source_before.tex').read_bytes()),
                source_after_sha256=sha(SOURCE.read_bytes()),
                method='Crop unchanged vector figures from the verified prior PDF; no additional document compiled.',
                remaining_live_figure='medsam3-workflow', figures=records)
(AUDIT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(figures_cached=len(records), bytes_before=len((AUDIT / 'source_before.tex').read_bytes()),
                      bytes_after=len(SOURCE.read_bytes()), assets_bytes=sum(r['bytes'] for r in records)), indent=2))
