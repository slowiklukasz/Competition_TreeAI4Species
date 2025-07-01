## TreeAI4Species Competition: Semantic Segmentation Solution

### Author: Łukasz Słowik (lukslow)

#### Method Overview

This solution achieves high-performance semantic segmentation through a 4-model weighted ensemble. The approach combines different architectures and backbones to maximize predictive accuracy and robustness.

The final inference pipeline integrates predictions from all four models and enhances them with Test-Time Augmentation (TTA) and a confidence threshold to improve precision.

#### 1. Model Ensemble Composition:

- Model A DeepLabV3 + efficientnet-b5 encoder, trained with DiceLoss and CrossEntropyLoss with light augmentations.
- Model B DeepLabV3 + efficientnet-b5 encoder, trained with with DiceLoss and CrossEntropyLoss with strong augmentations.
- Model C Unet++ + efficientnet-b5 encoder, trained with Dice+Focal loss.
- Model D DeepLabV3+ mit_b4 (Transformer) encoder. Best-performing single model, leveraging a Transformer backbone.

#### 2. How to Reproduce Predictions

This project was developed and tested in the Google Colab environment to ensure maximum reproducibility. The notebook handles all dependency installations automatically.

- Step 1: Open inference_semantic_segmentation.ipynb in Google Colab

        - Important: When prompted, make sure to select a GPU-accelerated runtime (Runtime > Change runtime type > T4 GPU) for the inference to work correctly and efficiently.

- Step 2: Prepare Data and Models

        - Place Model Weights: Download the four model checkpoints (.pth files) from my Google Drive (https://drive.google.com/drive/folders/1Kd1XOg3JndAocSsufb4BQ7ccFzpjKyo-?usp=sharing ) and place them in a location accessible from your Google Drive (e.g., in a folder named competition_weights).

        - Place Test Data: Place the test images in a separate folder on your Google Drive (e.g., competition_test_data).

- Step 3: Run the Notebook

        - Run the first few cells to mount your Google Drive and install the required packages.

        - Edit the configuration cell at the top of the notebook to set the correct paths to your test images and model weights on Google Drive (point number 3 in inference notebook).

        - Run all subsequent cells (Runtime > Run all).

        - The notebook will automatically process all images and save the final prediction masks as .npy files in the specified output directory on your Google Drive.
