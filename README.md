# YOGO-Net: A Variety-Aware Dual-Depth Fusion Pipeline for Non-Destructive Lettuce Biomass Estimation

We present **YOGO-Net (You Only Grow Once)**, a non-destructive computer vision framework designed to estimate the dry weight of various lettuce cultivars using multi-modal RGB-D sensor data.

Standard depth-to-weight models often struggle with sensor-to-target variance and morphological diversity.

Our framework introduces innovations to overcome these complexities and improve dry weight prediction accuracy (as measured by MAE) beyond the current SOTA:

🥬 **Dual-Domain Depth ($D^2$) Normalization**: Unlike traditional approaches that use a single global or local scale, we bifurcate the depth manifold. By anchoring to the crate-floor (_Global Volumetric Path_) while simultaneously stretching local leaf textures (_Local Morphological Path_), we resolve the ambiguity between plant size and leaf density.

🥬 **Semantic-Guided Geometric Rectification**: Rather than relying on fragile color-thresholding heuristics common in agricultural literature, we utilize high-fidelity U-Net segmentation to isolate biological mass. This "smart-cropping" protocol effectively eliminates background sensor noise and environmental artifacts, ensuring the model only reasons over valid biomass.

🥬 **Variety-Conditioned Routing**: We implement a "Variety Router" that performs real-time cultivar identification. This allows the system to act as a logic gate, directing the 5-channel feature cube (RGB + $D_{global}$ + $D_{local}$) into specific weights optimized for the unique morphology of the identified variety.

🥬 **Specialist Ensemble Orchestration**: Final predictions are generated via a Deep Ensemble of specialists. This mitigates model variance and leverages the decorrelated error patterns of 5-fold cross-validation, providing a robust "Council of Experts" for precise biomass estimation.

---

## Application

We demonstrate its application using the 3rd Autonomous Greenhouse Challenge: Online Challenge Lettuce Images dataset[^1], which has been divided into Training (231 samples), Test (76 samples) and Final (81 samples) sets.

### Preparatory scripts

| Script                                                                                               | Explanation                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [01_crop_raw_images_to_800.ipynb](scripts/01_crop_raw_images_to_800.ipynb)                           | Consumes raw 1920x1080 RGB-D image pairs from `data/raw/Training/` and applies a chroma-keyed region-of-interest (ROI) extraction protocol to extract an 800x800 crop centred on the biological mass. Cropped images are saved to `data/processes/crops/Training/`.'                                                                                                               |
| [02_train_lettuce_and_crate_masks_model.ipynb](scripts/02_train_lettuce_and_crate_masks_model.ipynb) | Trains a U-Net segmentation model to identify the lettuce mass and crate the lettuce is positioned upon. Ground-truth masks annotated for 70 images using CVAT[^2] with Segment Anything 2.0[^3]. Labels are stored within `data/labels/`. These segmentation masks are later used in developing the dual-domain depth representation. Model checkpoints are stored in `weights/`. |
| [03_infer_lettuce_and_crate_masks.ipynb](scripts/03_infer_lettuce_and_crate_masks.ipynb)             | Applies the trained segmentation model to infer lettuce and crate masks for the balance of the training image set. Masks are stored in `data/processes/masks_inferred/`.                                                                                                                                                                                                           |
| [04_ma_preprocess_with_mask_480.ipynb](scripts/04_ma_preprocess_with_mask_480.ipynb)                 | Performs two critical preprocessing activities. First, generates _local_ and _globally_ normalised depth maps, representing the core of the Dual-Domain Depth ($D^2$) approach. Second, applies significant geometric, photometric and noise augmentation. Consolidated and augmented image-pairs (RGB and depth variants) are stored in `data/augmented/`.                        |
| [05_train_variety_classifier.ipynb](scripts/05_train_variety_classifier.ipynb)                       | Trains an SE-ResNet-18 classifier to identify cultivar. Achieves 100% accuracy on validation (subset of training) set.                                                                                                                                                                                                                                                             |
| [06_train_dry_weight_regressor.ipynb](scripts/06_train_dry_weight_regressor.ipynb)                   | Trains a lettuce dry-weight regressor using a Multi-model Integration Fusion (MIF) architecture, allowing the organic features of the RGB stream to be contextually weighted by the volumetric signals of the $D^2$ maps. Training is undertaken across 5 folds using cultivar and dry weight (ground-truth) stratification. Model checkpoint are stored in `weights/`.            |

Due to file size constraints, the `data/raw/`, `data/processed/` and `weights/` folders are not available in this repo.

### Prediciton/inference scripts

Prediction is undertaken by holistic scripts for either the Test or Final image sets, where the relevant script undertakes all image pre-processing (including cropping, masking, resizing), cultivar inference and dry weight inference as a single pipeline. Predictions are output as Comma-Separated Value (CSV).

| Script                                                                                                     | Image Set | Prediction File | MAE    |
| ---------------------------------------------------------------------------------------------------------- | --------- | --------------- | ------ |
| [07_predict_dry_weight_Final.ipynb](scripts/07_predict_dry_weight_Final.ipynb)                             | Final     | final.csv       | 0.5534 |
| [07*predict_dry_weight_Final_calibrated*+1.ipynb](scripts/07_predict_dry_weight_Final_calibrated_+1.ipynb) | Final     | final.csv       | 0.4787 |
| [07*predict_dry_weight_Final_calibrated*-1.ipynb](scripts/07_predict_dry_weight_Final_calibrated_-1.ipynb) | Final     | final.csv       | 0.5905 |
| [07_predict_dry_weight_Test.ipynb](scripts/07_predict_dry_weight_Test.ipynb)                               | Test      | test.csv        | 0.6194 |

MAE reporting in the above table is as reported by the AIML Grand Challenge leaderboard.

---

### References

[^1]
Hemming, S. (S., de Zwart, H. F. (F., Elings, A. (A., bijlaard, monique, Marrewijk, van, B., & Petropoulou, A. (2021). 3rd Autonomous Greenhouse Challenge: Online Challenge Lettuce Images (Version 1) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/15023088.V1

[^2]
CVAT.ai Team. (2026). Computer Vision Annotation Tool (CVAT) [Software]. Available from https://github.com/cvat-ai/cvat

[^3]
Ravi, N., et al. (2025). SAM 2: Segment Anything in Images and Videos. International Conference on Learning Representations (ICLR).
