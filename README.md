# resnet50-transfer-learning-scene-classification
ResNet50 Transfer Learning for Scene Classification

This project applies **transfer learning** with ResNet50 to classify outdoor scene images from the Intel Image Classification dataset into six categories: **buildings, forest, glacier, mountain, sea, and street**.

The notebook demonstrates a full deep learning workflow in TensorFlow/Keras, including dataset loading, preprocessing with `preprocess_input`, model training, fine-tuning, saving trained weights, and interpreting results using training and validation plots.

## Project Overview

This project was built to explore how a pretrained convolutional neural network can be adapted to a new image classification task with limited training effort. Instead of training a CNN from scratch, the notebook uses **ResNet50 pretrained on ImageNet** as a feature extractor, then adds custom classification layers for the Intel scene dataset.

The project also includes a separate comparison experiment without `preprocess_input` to show how proper preprocessing affects convergence and model performance.

## Features

- Uses **ResNet50** as a pretrained backbone
- Applies **transfer learning** for scene classification
- Uses TensorFlow/Keras preprocessing with `preprocess_input`
- Includes **fine-tuning** after initial training
- Saves trained model **weights** for reuse
- Visualizes **training vs validation accuracy and loss**
- Includes a **separate preprocessing experiment** to compare convergence with and without `preprocess_input`

## Dataset

This project uses the **Intel Image Classification** dataset, which contains natural scene images grouped into six classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The dataset is downloaded directly in the notebook using `kagglehub`.

## Model Architecture

The model is built using:

- **ResNet50** with `include_top=False`
- Global feature extraction from pretrained ImageNet weights
- Custom classification head:
  - `Flatten()`
  - `Dense(512, activation='relu')`
  - `Dense(6, activation='softmax')`

During the first training phase, the pretrained ResNet50 layers are frozen. In the fine-tuning phase, selected deeper layers are unfrozen so the model can adapt better to the target dataset.

## Workflow

The notebook follows this pipeline:

1. Install required libraries
2. Download and extract the dataset
3. Load training and validation images
4. Apply `preprocess_input` for ResNet50
5. Build the transfer learning model
6. Train the model
7. Save trained weights
8. Reload the model architecture and weights
9. Fine-tune the deeper layers
10. Plot training and validation results
11. Run a separate experiment **without** preprocessing for comparison

## Why `preprocess_input` Matters

ResNet50 was originally trained on ImageNet with a specific input preprocessing scheme. Because of that, using `preprocess_input` is important for stable learning and better convergence.

This project includes a separate experiment without preprocessing to compare validation accuracy and observe how performance changes when preprocessing is omitted.

## Results Interpretation

The notebook includes plots to help answer two key questions:

### 1. Preprocessing Check
A separate experiment is run without `preprocess_input` to compare performance against the properly preprocessed model. This helps show whether the model converges more poorly or achieves lower validation accuracy without the expected ResNet50 input normalization.

### 2. Overfitting Check
Training and validation accuracy/loss curves are plotted to determine whether the model is:

- learning to generalize well, or
- starting to memorize the training set

If training accuracy keeps increasing while validation accuracy stalls or drops, the model may be overfitting. If both improve together and remain reasonably close, the model is generalizing better.

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Google Colab**
- **Matplotlib**
- **NumPy**
- **PIL**
- **KaggleHub**

## File Structure

A possible repository structure:

```text
resnet50-transfer-learning-scene-classification/
├── README.md
├── Transfer Learning.ipynb
├── requirements.txt
└── results/
    ├── training_validation_plot.png
    └── preprocessing_comparison_plot.png
```

## How to Run

### In Google Colab
1. Open the notebook in Google Colab.
2. Run all cells in order.
3. Wait for the dataset to download and training to complete.
4. View the generated plots for training/validation behavior and preprocessing comparison.

### Notes
- If you only want to test the notebook structure, reduce the number of epochs temporarily.
- For the final run, use your intended training and fine-tuning epoch settings.
- The notebook saves **weights only**, not a full serialized `.keras` model, due to serialization issues with internal wrapped layers.

## Saving and Reloading

This project saves the trained model using:

```python
resnet_model.save_weights("resnet_transfer.weights.h5")
```

To use the trained model later, rebuild the same architecture and load the saved weights:

```python
resnet_model.load_weights("resnet_transfer.weights.h5")
```

## Practical Applications

Although this project is educational, it demonstrates a workflow that is practical for real-world computer vision tasks such as:

- Scene recognition
- Image auto-tagging
- Environmental image labeling
- Travel or location photo categorization
- Transfer learning baselines for small image datasets

It can also be used as a starting template for other image classification projects using pretrained CNNs.

## Possible Improvements

Future improvements could include:

- Data augmentation
- More balanced hyperparameter tuning
- Better experiment tracking
- Confusion matrix and per-class evaluation
- Export to deployment-friendly formats
- Comparison with other pretrained backbones such as EfficientNet or MobileNet

## Author

**Steven Flogio**

Computer science / IT student focused on AI/ML, computer vision, and practical deep learning workflows.

## License

This project is for educational and portfolio use. Add a license such as MIT if you plan to make the repository publicly reusable.
