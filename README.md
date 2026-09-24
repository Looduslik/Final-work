```markdown
# Image Classification for Clothing Items

## Project Overview
This project focuses on building and evaluating deep learning models for image classification of clothing items into two categories: 'bluzy' (blouses) and 'bryuki' (trousers). The goal is to accurately predict the class of a single clothing item image on a plain background. This task was part of a demo exam for an 'Applied Data Analysis' course, referencing a Kaggle competition: [LaModa Images Classification](https://www.kaggle.com/competitions/lamoda-images-classification).

## Dataset
The dataset consists of images of blouses and trousers. Key characteristics and preprocessing steps include:
- Data Source: Kaggle competition dataset, downloaded directly into the Colab environment.
- Image Resolution: Initial exploration revealed varying image resolutions (e.g., 46x66 and 600x866). All images were resized to 224x224 pixels for consistency before model input.
- Class Balance: Class labels were extracted from filenames, and the dataset was split into training and validation sets with stratification to maintain class balance.
- Augmentation: Standard image transformations including resizing, random horizontal flip (for training), converting to PyTorch tensors, and normalization using ImageNet statistics were applied.

## Methodology
Three different convolutional neural network (CNN) models were implemented and compared:

1.  RSNAModel (Custom CNN): A custom-built CNN architecture with three convolutional blocks (each followed by BatchNorm, ReLU, and MaxPool) and two fully connected layers. This model was trained from scratch.
2.  ResNet18 (Pre-trained): A pre-trained ResNet18 model (on ImageNet) where only the final fully connected layer was replaced and fine-tuned for binary classification.
3.  VGG16 (Pre-trained): A pre-trained VGG16 model (on ImageNet) with its classifier's last layer adapted for the binary classification task.

### Training Details
- Device: Training was performed on a GPU (CUDA) when available.
- Loss Function: `nn.CrossEntropyLoss` was used.
- Optimizer: `Adam` optimizer with a learning rate of 0.001.
- Epochs: Each model was trained for 10 epochs.
- Mixed Precision: `torch.cuda.amp.autocast` and `GradScaler` were utilized for mixed-precision training to enhance speed and reduce memory consumption.
- Evaluation: Models were evaluated on a validation set after each epoch, and the best model state (based on validation accuracy) was saved.

## Results and Conclusion

All three models demonstrated high performance on the binary classification task. The key findings are:

   RSNAModel (Custom CNN): Achieved high accuracy (around 99%) and showed good learning capability, indicating that a custom-built, relatively simple architecture can be very effective with proper tuning.
   ResNet18 (Pre-trained): Showed very fast convergence and stable, high accuracy (above 99%). This highlights the significant benefits of transfer learning, where the model leverages powerful pre-learned features from a large dataset like ImageNet.
   VGG16 (Pre-trained): Also achieved excellent, stable performance (above 99%) comparable to ResNet18. While effective, VGG models typically have more parameters than ResNet models, potentially making them more computationally demanding.

Overall Comparison:
Pre-trained models (ResNet18, VGG16) generally offered faster convergence and more stable metrics compared to the custom-trained RSNAModel. For this specific task, ResNet18 with pre-trained weights appeared to be the most balanced and effective solution, providing high accuracy with reasonable computational resource requirements. The project concluded with generating submission files for Kaggle for each trained model.
```
