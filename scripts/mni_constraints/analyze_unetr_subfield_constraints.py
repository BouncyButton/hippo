"""Apply the shared archive checks to original SDF/depth candidates."""
from scripts.mni_constraints import analyze_unetr_window_study as analysis
from scripts.mni_constraints import unetr_subfield_constraints_study as study


def main():
    analysis.study=study
    analysis.RESULT_FILENAME='subfield_results.json';analysis.PLOT_STEM='subfield_results'
    analysis.TITLE='UNETR original SDF/depth constraints: gain vs matched baseline\n* development pilot; blank cells not run'
    analysis.SCOPE='Original SDF/depth definitions, unchanged409-update53.305T base recipe. Fixed calibration targets5% or20%; no union bands/edge; envelope1 throughout.'
    analysis.main()


if __name__=='__main__':main()
