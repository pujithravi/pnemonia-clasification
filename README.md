# Explainable AI for Pneumonia Classification

A CNN that classifies chest X-rays as **NORMAL** or **PNEUMONIA**, with **LIME** and **SHAP** used to show which regions of the X-ray drove each prediction.

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) – 5,216 training images and 624 test images across 2 classes (`NORMAL = 0`, `PNEUMONIA = 1`).

## Pipeline
1. **Preprocessing** – `ImageDataGenerator`: rescale to [0, 1], resize to 300×300, augmentation (shear 0.2, zoom 0.2, horizontal flip).
2. **Model** – custom CNN: 5 × (Conv2D 3×3 + ReLU → MaxPool 2×2) with 16 → 32 → 64 → 128 → 128 filters, then Flatten → Dense(256) → Dense(512) → Dense(1, sigmoid). About 1.98M parameters.
3. **Training** – Adam optimizer, binary cross-entropy loss.
4. **Explainability**
   - **LIME** (`LimeImageExplainer`, quickshift segmentation, up to 5,000 perturbation samples): highlights the superpixels that push the prediction toward or away from pneumonia.
   - **SHAP** (`DeepExplainer` with training images as background): pixel-level attributions shown with `shap.image_plot`.

## Results
This is a baseline, and the focus is explainability. After a short 2-epoch training run, the model reaches about **64% test accuracy**.

## Planned improvements
- Transfer learning (ResNet50 / DenseNet121) instead of a CNN trained from scratch
- Class weights to handle the NORMAL/PNEUMONIA imbalance
- A proper held-out validation split, longer training with early stopping
- Report precision, recall, F1 and a confusion matrix (recall matters most in a medical setting)
- Grad-CAM heatmaps alongside LIME and SHAP

## Run
```bash
pip install numpy matplotlib opencv-python tensorflow lime shap scikit-image
jupyter notebook "main (1).ipynb"
```
Update the dataset paths in the notebook to point to your local `chest_xray/` folder.
