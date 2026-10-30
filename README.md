# r01-imd-placement-sim

Simulation code supporting the preliminary studies of the R01 renewal
"Adaptive Robotic Placement of Implantable Microdevices for In-Tumor Drug Testing in Prostate Cancer"
(Nobuhiko Hata, Brigham and Women's Hospital / Harvard Medical School; renewal of R01EB030539).

The planner learns each patient's needle deflection from the devices already placed
(Gaussian process, squared-exponential kernel, length scale 10 mm) and re-plans the
remaining devices. The static arm uses the same planner without updates. Both arms see
the same simulated deflection field and noise for each simulated patient.

| Script | What it does |
|---|---|
| `mc_adaptive_microdevice.py` | P2 simulation: four devices in a circular 2D lesion (10 or 14 mm), deflection offset up to 3.5, 6, or 9 mm with a weak spatial gradient, per-insertion noise 0.4 or 1.5 mm, 300 simulated patients per condition. Reports pattern error and share of devices inside the lesion. |
| `covering_ratio.py` | Covering ratio (largest lesion-to-nearest-device distance / equal-area radius) for the P2 simulation; sets the 0.9 spread target. |
| `lesion_sim_3d.py` | 3D version on the same 220 PI-CAI lesions. Each IMD is a 6.5 mm x 0.75 mm device along the needle path (cranio-caudal), with 20 reservoirs at 7 levels; depth error is added; lesions of <=10, 10-15, and >15 mm (largest dimension) receive 2, 3, or 4 devices, following Rainer et al. (J Magn Reson Imaging 2025). Reports the reservoir-in-target (RIT) fraction by lesion size, static vs adaptive, and the ceiling under error-free placement. |
| `figures/p2_sim_3d_figure.py` | Figure 8 E-F of the application: example 12 mm lesion and RIT by lesion size at a 6 mm offset. |
| `anatomic_lesion_sim.py` | Same planner on real lesion anatomy: the largest axial cross-section of 220 expert-delineated csPCa lesions from the public PI-CAI dataset, i.e., the plane perpendicular to a transperineal needle. Target slots are the centroids of a 4-region centroidal Voronoi (k-means) partition of each lesion. |
| `fetch_picai_lesions.sh` | Downloads the PI-CAI expert lesion masks (CC BY-NC 4.0) into `data/` (not tracked). |
| `power_calculations.py`, `swine_baseline.py` | Sample-size and baseline calculations used in the Approach. |

## Anatomic results (`results/anatomic_lesion_sim_N100.txt`, 100 simulated patients per lesion and arm)

220 lesions; axial equivalent diameter median 13.5 mm (IQR 10.1–17.0), 24% ≤ 10 mm.
Ideal four-slot covering ratio median 0.69.

| Offset (mm) | Noise (mm) | Devices in lesion, static → adaptive | Lesions ≤ 10 mm | Pattern error (mm) | Covering ratio ≤ 0.9 |
|---|---|---|---|---|---|
| 2.5 | 0.4 | 89.5 → 97.1% | 71.7 → 91.9% | 2.1 → 1.5 | 54.8 → 78.5% |
| 2.5 | 1.5 | 79.3 → 83.5% | 60.3 → 67.4% | 2.9 → 2.6 | 30.0 → 35.3% |
| 3.5 | 0.4 | 78.4 → 94.0% | 54.1 → 85.9% | 2.8 → 1.8 | 31.6 → 60.5% |
| 3.5 | 1.5 | 72.3 → 81.4% | 51.0 → 64.7% | 3.4 → 2.8 | 19.9 → 28.5% |
| 6 | 0.4 | 53.5 → 83.6% | 29.2 → 69.5% | 4.6 → 2.9 | 7.0 → 24.8% |
| 6 | 1.5 | 53.2 → 73.0% | 29.4 → 52.8% | 5.0 → 3.6 | 6.0 → 13.1% |
| 9 | 0.4 | 35.0 → 69.5% | 11.5 → 43.4% | 6.7 → 4.4 | 1.4 → 6.8% |
| 9 | 1.5 | 35.5 → 60.9% | 12.7 → 34.1% | 7.0 → 4.9 | 1.4 → 4.1% |

Adaptive planning gave lower pattern error than the static plan in 73–100% of simulated patients.

## 3D results (`results/lesion_sim_3d_N30.txt`, 30 simulated patients per lesion and arm)

Largest lesion dimension 6.0-60.0 mm (median 18.7); cranio-caudal extent 3.0-60.0 mm.
Reservoirs in the lesion (RIT, %), static -> adaptive, at a 6 mm offset and 0.4 mm noise:

| Largest dimension (mm) | Lesions | Static -> adaptive | Error-free ceiling |
|---|---|---|---|
| <=10 | 17 | 9.7 -> 37.2 | 72.7 |
| 10-15 | 43 | 31.3 -> 63.7 | 87.3 |
| 15-20 | 64 | 46.1 -> 74.5 | 93.6 |
| 20-27 | 55 | 57.6 -> 81.3 | 97.1 |
| >27 | 41 | 81.7 -> 90.7 | 98.3 |
| all | 220 | 49.9 -> 74.2 | |

In lesions of 10 mm or less, the median cranio-caudal extent (6 mm) barely exceeds the 6.5 mm device,
which caps RIT even under error-free placement; the planner gains most in lesions of 10-27 mm.
Results for offsets of 2.5 and 9 mm and noise of 1.5 mm are in the results file.

## Run

```
pip install numpy scipy scikit-learn nibabel scikit-image matplotlib
python mc_adaptive_microdevice.py
./fetch_picai_lesions.sh
python anatomic_lesion_sim.py data/picai_labels/csPCa_lesion_delineations/human_expert/resampled 100
python lesion_sim_3d.py data/picai_labels/csPCa_lesion_delineations/human_expert/resampled 30
python figures/p2_sim_3d_figure.py data/picai_labels/csPCa_lesion_delineations/human_expert/resampled
```

## Limitations

`mc_adaptive_microdevice.py` and `anatomic_lesion_sim.py` are 2D didactic models of the needle-perpendicular plane; `lesion_sim_3d.py` adds device length and depth error but treats each reservoir as a point and uses the MRI slice spacing (about 3 mm) for the cranio-caudal axis. Deflection is a synthetic
linear field, not a tissue model, and lesion masks come from MRI delineation, not histology.

## Data credit

Lesion masks: PI-CAI (Saha A, Bosma JS, et al. Lancet Oncol 2024;25(7):879–887,
doi:10.1016/S1470-2045(24)00220-1), https://github.com/DIAGNijmegen/picai_labels, CC BY-NC 4.0.
