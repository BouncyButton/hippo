"""Configure the archive-gated matrix for a verified reduced-peak-LR cosine UNETR winner."""
from scripts.mni_constraints import verified_unetr_full_matrix as matrix
from scripts.mni_constraints import unetr_lower_lr_study as confirmation
from scripts.mni_constraints import verified_lower_lr_matrix_recipe as recipe_helper

matrix.confirmation=confirmation
matrix.recipe_helper=recipe_helper
matrix.NAME='mni_verified_unetr_lower_lr_constraints14_53t_0519_20261109'
matrix.MODULE='scripts.mni_constraints.verified_lower_lr_full_matrix'
matrix.RESULT_FILENAME='lower_lr_results.json'
matrix.PROFILE_FILENAME='dropout0_original_constraints.json'
matrix.RUNTIME_FILENAME='unetr_lower_lr_runtime.py'
matrix.ADD_MODULES=(*matrix.ADD_MODULES,'verified_lower_lr_full_matrix.py','verified_lower_lr_matrix_recipe.py')

if __name__=='__main__':matrix.main()
