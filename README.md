# Corrosion Detection Using Deep Learning


Below is a template for another sample project. Please follow this template.
# [Deep Learning Project Template] A Computer Vision-Based Framework for Industrial Corrosion Detection

## Introduction
This study addresses the challenge of detecting corrosion effectiveness on industrial surfaces, a critical task for ensuring structural safety and reducing maintenance costs. The research focuses on binary image classification using computer vision and advanced deep learning techniques. It evaluates the performance of Convolutional Neural Network (CNN) based ResNet models in three variants: ResNet18, ResNet50, and ResNet101, with integrated architectural enhancements. The models are trained using transfer learning on the labeled Phase5 Capstone Project dataset. The proposed approach implements a modified focal loss function, which help to improve classification accuracy and address class imbalance.  Additionally, an attention block is incorporated into the network architecture to support the model focus on important regions within the image, enhancing feature extraction. The enhanced model improves performance, robustness, and generalization compared to the baseline models. 

## Project Metadata

### Author
- **Name:** Bayan Aldahlawi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliation:** KFUPM

### Project Documents
- **Code Implementation:** [Jupyter Notebook](/Bayan%20Aldahlawi%20-%20A%20Computer%20Vision-Based%20Framework%20for%20Industrial%20Corrosion%20Detection.ipynb)
- **Term Paper:** [Project Report](/Bayan%20Aldahlawi%20-%20A%20Computer%20Vision-Based%20Framework%20for%20Industrial%20Corrosion%20Detection.pdf)
- **Presentation:** [Project Presentation](/Bayan%20Aldahlawi%20-%20A%20Computer%20Vision-Based%20Framework%20for%20Industrial%20Corrosion%20Detection.pptx)

### Reference Paper
- [A Deep Learning Approach to Industrial Corrosion Detection](https://www.sciencedirect.com/org/science/article/pii/S154622182400777X)

### Reference Dataset
- [Phase 5 Capstone Project Dataset](https://github.com/pjsun2012/Phase5_Capstone-Project)



## Project Technicalities

### Terminologies
- **Convolutional Neural Network (CNN):** A deep learning architecture designed to process and analyze spatial data, particularly images.
- **ResNet Architecture:** A CNN variant that uses residual connections (skip connections) to enable training of very deep networks without degradation.
- **Focal Loss:** A loss function that down-weights easy examples and focuses learning on hard-to-classify examples, especially useful for class-imbalanced datasets.
- **Attention Mechanism (MHSA):** Multi-Head Self-Attention (MHSA) focuses on important regions in feature maps, enhancing feature extraction by modeling global dependencies.
- **Data Augmentation:** A technique that generates during training samples by applying transformations such as flipping, rotation, and scaling, to increase the dataset.
- **Transfer Learning:** Using a pretrained model (such as ImageNet-trained ResNet) and fine-tuning it for a specific task like corrosion classification.


### Problem Statements
- **Problem 1:** Corrosion datasets are small and imbalanced, leading to overfitting and biased learning.
- **Problem 2:** Conventional loss functions (e.g., binary cross-entropy) fail to handle class imbalance effectively.
- **Problem 3:** Models may focus on irrelevant background regions rather than actual corrosion areas, reducing accuracy.

### Research Gaps Addressed
- **Handling Class Imbalance:** Standard models often misclassify minority corrosion cases.
- **Feature Focus:** Lack of attention mechanisms results in weak feature extraction from corrosion-affected areas.
- **Overfitting Risk:** Small datasets lead to overfitting without proper augmentation and regularization strategies.

### Proposed 3 Ideas to Solve the Problems
1. **Integrating Focal Loss:** Improve model robustness by focusing more on hard-to-classify corrosion samples.
2. **Embedding Attention Blocks (MHSA):** Enhance spatial feature extraction by guiding the model to important corrosion regions.
3. **Extensive Data Augmentation:** Expand dataset variability to improve generalization and reduce overfitting risk.

### Proposed Solution: Code-Based Implementation
This repository provides a custom implementation based on PyTorch and keras, featuring:

- **Modified ResNet Architectures:** ResNet18, ResNet50, and ResNet101 integrated with Focal Loss and Attention Blocks (MHSA).
- **Advanced Loss Handling:** Modified focal loss replaces standard binary cross-entropy to address class imbalance.
- **Optimized Training Strategy:** Transfer learning with pretrained ImageNet weights, coupled with data augmentation techniques, to ensure better corrosion classification across various industrial surfaces.


### Key Components
- **`model`**: Contains the modified ResNet architectures (ResNet18, ResNet50, ResNet101) integrated with Focal Loss and Attention Blocks (MHSA).
- **`train`**: Script to handle model training, loss calculation, and optimization with configurable hyperparameters.
- **`utils`**: Utility functions for data loading, preprocessing (augmentation), evaluation metrics (accuracy, loss curves).
- **`inference`**: Script for model evaluation and prediction on test images for corrosion detection.

## Model Workflow
The workflow of the Enhanced ResNet-based Corrosion Detection Model is designed to classify industrial surfaces into corroded and non-corroded categories through deep feature extraction and improved loss handling:

1. **Input:**
   - **Image Input:** The model takes a 2D image of an industrial surface (e.g., pipeline, metal plate) as input.
   - **Preprocessing:** Images are resized, normalized (ImageNet statistics), and augmented (flips, rotations) to improve robustness during training.

2. **Feature Extraction:**
   - **Backbone Network:** Images are passed through a pretrained ResNet (18, 50, or 101), where spatial features are extracted.
   - **Attention Enhancement:** A Multi-Head Self-Attention (MHSA) block focuses on important regions within feature maps, reducing background noise and improving critical feature representation.

3. **Classification Head:**
   - **Fully Connected Layers:** After feature extraction, a classification head processes the output to predict the probability of corrosion.
   - **Loss Handling:** A modified Focal Loss is used to address class imbalance by emphasizing learning from hard-to-classify corrosion cases.

4. **Output:**
   - **Prediction:** The final model outputs a binary prediction — corroded or non-corroded — for each input image.
   - **Evaluation:** Model performance is assessed using accuracy, and training/validation loss curves.


## How to Run the Code

## How to Run the Project

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/Corrosion-Detection-Using-Deep-Learning.git
    cd Corrosion-Detection-Using-Deep-Learning
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Train the ResNet model (ResNet18, ResNet50, or ResNet101) with Focal Loss and Attention Block enhancements.
    ```bash
    python train.py --model resnet50 --epochs 100 --batch_size 32 --learning_rate 0.0001
    ```
    *(You can change `--model` to `resnet18` or `resnet101` based on the variant you want to train.)*

4. **Evaluate the Model:**
    After training, evaluate the model performance on validation or test datasets.
    ```bash
    python inference.py --checkpoint path/to/best_model.pth --input_folder path/to/test_images/
    ```

---

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, keras, and other libraries that made this project possible.
- **Supervision:** Special thanks to Dr. Muzammil Behzad for his invaluable guidance and continuous support throughout the project.
- **Resource Providers:** Gratitude to KFUPM and supporting platforms for providing the computational resources necessary for training and evaluation.


