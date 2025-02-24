# Custom Kazakh Language with EasyOCR

This repository contains an attempt to train [EasyOCR](https://github.com/JaidedAI/EasyOCR/tree/master) on the Kazakh language using Google Colab. The goal was to fine-tune an EasyOCR model and compare it with a pretrained Russian model. Didn't spend much of time on it, but I figured that EasyOCR doesn't have a Kazakh language in its dataset, and I wonder why, that's why I tried to do it.

!Note, easyOCR can accept new language if one provided a proper dataset:

> **To request a new language**, we need you to send a PR with the following two files:  
> - In folder `easyocr/character`, we need **`yourlanguagecode_char.txt`** that contains a list of all characters. Please see format examples from other files in that folder.  
> - In folder `easyocr/dict`, we need **`yourlanguagecode.txt`** that contains a list of words in your language. On average, we have **~30,000 words** per language, with more than **50,000 words** for more popular ones. More is better in this file.  



## Overview

1. **Dataset Creation**:  
   - Generated a custom dataset of **2000 images**, split into training and validation sets.
  
2. **Model Training**:  
   - Trained a new EasyOCR model on the Kazakh language using **default settings**.
   - Fine-tuned the **pretrained Russian model** on the Kazakh dataset.

3. **Inference & Results**:  
   - Evaluated both models on a single test image.
   - Neither model performed well (as expected), but the **fine-tuned Russian model** showed slightly better results.

4. **Future Improvements**:
  - Increase dataset size and diversity. Include proper methods for processings.
  - Optimize training settings (augmentations, hyperparameters, better choice of models for detection and recognition).
  
Step by step implementation is in .ipynb file


