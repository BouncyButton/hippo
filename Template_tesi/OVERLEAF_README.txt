FAST-COMPILE THESIS PACKAGE

Upload the ZIP as a project, or upload all its files into your existing project.
Set the main document to thesis-revision.tex and use pdfLaTeX.
If your existing project calls its main file document.tex, you may replace that
file with thesis-revision.tex's contents; keep the figures folder alongside it.

IMPORTANT: the TeX file requires figures/compiled/*.pdf and
figures/qualitative/*.png. Copying only the
TeX text is insufficient. No shell escape, TikZ externalization build or extra
bibliography tool is required: references remain embedded in this source.

Static figures are loaded from PDF assets, preserving their original vector
text, paths and native image pixels. One small adaptation workflow remains live
TikZ because its wording was revised after the preserved PDF was generated.
Captions, labels, cross-references and the contents remain ordinary LaTeX.

Two PNG figures show the verified qualitative reproductions: typical cases in
Results and a consolidated adverse-case figure in the appendix. These are pre-rendered
assets and add no TikZ drawing work. Their scores are separate from the original
aggregate results, as explained in the text and supplementary comparison table.

The editable TikZ for the cached figures is retained in figures/source/*.tikz.
These files are not read during normal compilation. If you change a diagram or
chart, update its compiled PDF as well; editing the saved TikZ alone will not
change the displayed figure. Ordinary prose or caption edits need no asset update.

The printed appendices have been consolidated. All six complete aggregate Dice
tables remain in the thesis, including negative outcomes and matched controls.
The accompanying thesis-supplementary-results.zip preserves full run records,
error counts, fold breakdowns, complementary Dice scores, extra panels and the
previous long appendix as a source fragment. It is an evidence supplement and
does not need to be unpacked to compile the thesis.

This is not draft mode: every figure referenced by the current thesis is included.
Actual compile time still depends on the Overleaf compiler, template and account limit.

This upload package is a snapshot of the thesis at the time it was generated.

The standalone abstract.txt is for separate submission and is not compiled into
the thesis. The opening sequence is two blank pages, acknowledgements, one blank
page, then the contents and main text. Add your personal acknowledgements in the
marked source location before submission.
