# Kaggle GPU Refinement

The runner warms the model cache first; a non-zero exit means
at least one system failed.

1. Build bundle:
   python scripts/kaggle/make_bundle.py --engine mace \
     --systems hcooh hcooh_h2o hcho_nh3 --out mace_bundle.zip
   To run one system on both GPUs, plan a split from an engine that
   already ran it (weights = its force calls per reaction) and pass it:
     PYTHONPATH=src python scripts/kaggle/plan_split.py --system cpd_ma \
       --weights-from uma --parts 2 --out data/cpd_ma/kaggle_split.json
     python scripts/kaggle/make_bundle.py --engine mace --systems cpd_ma \
       --split-file data/cpd_ma/kaggle_split.json --out mace_bundle.zip
   Each part runs in its own copy under work-dir/parts/ (part 0 also does
   the species); new output is joined into the main results before the
   zip is made, so resume and merge work as for one worker.
2. Upload bundle to Kaggle as a private dataset (e.g. mse-mace-bundle).
3. Create notebook from scripts/kaggle/refine_on_kaggle.ipynb:
   Set GPU T4 x2, Internet on, set BUNDLE_DIR, run Save & Run All.
4. If run hits deadline, download partial zip, upload as dataset,
   set RESUME_ZIP="/kaggle/input/<dataset>/mace_results.zip", and re-run.
5. Download mace_results.zip and merge results into repo:
   python scripts/kaggle/merge_results.py --zip mace_results.zip --engine mace
