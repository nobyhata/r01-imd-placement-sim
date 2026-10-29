#!/usr/bin/env bash
# Download the PI-CAI expert csPCa lesion delineations (CC BY-NC 4.0) used by anatomic_lesion_sim.py.
# Source: https://github.com/DIAGNijmegen/picai_labels  (Saha et al., Lancet Oncol 2024; Bosma et al.)
set -euo pipefail
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/DIAGNijmegen/picai_labels data/picai_labels
cd data/picai_labels
git checkout HEAD -- clinical_information/marksheet.csv
python3 - <<'PY'
import csv
rows=[r for r in csv.DictReader(open('clinical_information/marksheet.csv')) if r['case_csPCa']=='YES']
open('/tmp/picai_pos.txt','w').write('\n'.join(f"csPCa_lesion_delineations/human_expert/resampled/{r['patient_id']}_{r['study_id']}.nii.gz" for r in rows))
PY
git sparse-checkout init --no-cone
git sparse-checkout set --no-cone $(cat /tmp/picai_pos.txt) clinical_information/marksheet.csv
git read-tree -mu HEAD
echo "lesion masks: $(ls csPCa_lesion_delineations/human_expert/resampled | wc -l)"
