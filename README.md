# AI vs. Real Image Classifier

A Google Colab notebook for training an image classifier that distinguishes AI-generated (`FAKE`) images from real (`REAL`) images. The project uses a pretrained EfficientNetV2-S model with PyTorch and saves training progress to Google Drive.

## Live website and source code

### [Open the live AI vs. Real website →](http://r48ow0g404cc0og8s04o8csw.46.225.136.253.sslip.io/)

| Project | Repository | Purpose |
| --- | --- | --- |
| **Frontend** | **[DNeberize/ai-or-real](https://github.com/DNeberize/ai-or-real-front)** | Website interface for uploading images and viewing predictions |
| **Backend** | **[DNeberize/ai-vs-real-backend](https://github.com/DNeberize/ai-vs-real-backend)** | API that loads the trained model and processes prediction requests |
| **Model training** | **This repository** | Google Colab workflow used to train and export the classifier |

> **Complete project:** this notebook trains the model, the backend serves it through an API, and the frontend provides the public user interface.

## Important: this project runs in Google Colab

This notebook was written in Google Colab and is intended to run there. The current code depends on Colab-specific features:

- `google.colab.drive.mount()` for access to the dataset and saved checkpoints;
- `google.colab.files.upload()` for testing individual images;
- Colab paths such as `/content` and `/content/drive/MyDrive`;
- a Colab GPU runtime for practical training speed.

Because of these dependencies, the notebook will not run unchanged in a local Jupyter environment or directly on GitHub. GitHub stores and displays the notebook; Google Colab provides the runtime that executes it and authorizes access to Google Drive.

Google Drive can also be accessed outside Colab through the Google Drive API or other authentication methods, but that setup is not implemented in this project.

## Project structure

```text
ai_vs_real/
├── ai_vs_real.ipynb   # Training, evaluation, checkpointing, and inference
├── README.md             # Project documentation
└── .gitignore            # Excludes datasets, checkpoints, and local files
```

The dataset and trained model files are intentionally kept in Google Drive rather than in this GitHub repository.

## Dataset setup

Create this folder in Google Drive:

```text
My Drive/ai_vs_real_images/
```

Place the dataset archive here:

```text
My Drive/ai_vs_real_images/Compressed.zip
```

The ZIP archive should contain this structure. `TRAIN` is required; `VAL` and `TEST` are optional:

```text
Compressed.zip
├── TRAIN/
│   ├── FAKE/
│   └── REAL/
├── VAL/                  # Optional; otherwise 15% of TRAIN is used
│   ├── FAKE/
│   └── REAL/
└── TEST/                 # Optional, but recommended for final evaluation
    ├── FAKE/
    └── REAL/
```

An extra top-level folder inside the ZIP is supported because the notebook automatically flattens one nested directory after extraction.

## How to run

1. Upload `Compressed.zip` to the Google Drive path shown above.
2. Open [`ai_vs_real.ipynb`](./ai_vs_real.ipynb) in Google Colab.
   - In Colab, select **File → Open notebook → GitHub** and enter this repository's URL.
   - You can also upload the notebook directly to Colab.
3. Select **Runtime → Change runtime type → GPU**.
4. Run the notebook cells from top to bottom.
5. Approve the Google Drive permission request when prompted.

The first model run may download pretrained EfficientNet weights. Standard Colab environments normally include the required packages: PyTorch, TorchVision, NumPy, Pillow, scikit-learn, and tqdm.

## What the notebook does

- mounts Google Drive and extracts the dataset into the temporary Colab runtime;
- caches dataset information to avoid unnecessary extraction;
- creates a stratified validation split when a `VAL` folder is not supplied;
- applies image augmentation and ImageNet normalization;
- fine-tunes an EfficientNetV2-S binary classifier;
- uses mixed-precision training when a GPU is available;
- tracks loss, accuracy, and macro F1 score;
- saves resumable checkpoints and stops early when validation F1 stops improving;
- evaluates the best checkpoint on `TEST`, when present;
- predicts an uploaded image as `AI-Generated` or `Real`.

## Main configuration

The main settings are near the beginning of the notebook:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `RESUME_FROM_LAST` | `True` | Continue the latest saved run |
| `TARGET_TOTAL_EPOCHS` | `20` | Total number of training epochs |
| `VAL_FRACTION` | `0.15` | Validation fraction when `VAL` is absent |
| `RANDOM_SEED` | `42` | Reproducible data split and training seed |
| `TRAIN_IMAGE_SIZE` | `224` | Input image size |
| `TRAIN_BATCH_SIZE` | `128` | Training batch size |
| `EVAL_BATCH_SIZE` | `256` | Validation and test batch size |
| `BACKBONE_NAME` | `efficientnet_v2_s` | Pretrained model architecture |

If Colab runs out of GPU memory, reduce `TRAIN_BATCH_SIZE` and `EVAL_BATCH_SIZE`.

## Saved files

Training artifacts are written to:

```text
My Drive/ai_vs_real_model_checkpoints/
```

The notebook creates files similar to:

```text
ai_vs_real_model_checkpoints/
├── best_model.pth
├── dataset_cache.json
├── latest_run.json
└── runs/<timestamp>/
    ├── best_checkpoint.pth
    ├── last_checkpoint.pth
    ├── epoch_metrics.csv
    ├── epoch_metrics.jsonl
    ├── run_config.json
    └── run_summary.json
```

With `RESUME_FROM_LAST = True`, rerunning the notebook continues from `last_checkpoint.pth`. Set it to `False` when you want to create a new training run.

## Inference

After training, run the inference cells at the end of the notebook. Colab will open a file picker, load the best available checkpoint from Google Drive, and print the predicted label and confidence score.

Model predictions are experimental and should not be treated as definitive proof that an image is real or AI-generated.
