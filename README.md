# Speech-Based Sentiment Analysis

This project involves downloading and preprocessing multiple speech emotion datasets, training a Convolutional Neural Network (CNN) to classify emotions, and mapping these emotions to sentiments (positive, neutral, negative).

## Datasets Used

1. [RAVDESS](https://zenodo.org/record/1188976)
2. [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
3. [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad)

## Dataset Preparation

- Audio files are downloaded and extracted.
- Files are organized into folders based on emotion categories: `angry`, `happy`, `sad`, `fear`, `neutral`, `disgust`, and `surprise`.
- Audio is preprocessed using MFCC features (Mel Frequency Cepstral Coefficients).

## Model Architecture

Two separate CNN models were trained:

### 1. Emotion Detection
- Classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`
- Input: 40x174 MFCC features
- Layers: Conv2D, MaxPooling2D, Dropout, Dense

### 2. Sentiment Classification
- Mapped emotions to sentiments:
  - `positive`: happy
  - `neutral`: neutral
  - `negative`: angry, sad, fear, disgust
- Output: `positive`, `neutral`, `negative`

## Results

- **Emotion Accuracy**: ~`39.42%%`
- **Sentiment Accuracy**: ~`65.55%%`
- Classification reports and confusion matrices included in the analysis.

## Dependencies

- `tensorflow`, `keras`, `librosa`, `sklearn`, `matplotlib`, `seaborn`, `tqdm`, `requests`