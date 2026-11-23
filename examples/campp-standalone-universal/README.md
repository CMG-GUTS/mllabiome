To run the training and inference first install all mllabiome dependencies (https://github.com/CMG-GUTS/mllabiome).

Then for each of the scripts, e.g.:
```bash
mllabiome examples/campp-standalone-universal/lodo__lampp_crc_cm_configs_sweep.py --stage evaluate

mllabiome examples/campp-standalone-universal/lodo__lampp_crc_cm_configs_sweep.py --stage base-selection

mllabiome examples/campp-standalone-universal/lodo__lampp_crc_cm_configs_sweep.py --stage inference
```

This results in end-to-end training, cross-validation evaluation on development dataset, and inference on external set with resulting predictions saved.

To package the predictions for leaderboard submission run:

```bash
python examples/campp-standalone-universal/package_campp_submission_clean.py
```