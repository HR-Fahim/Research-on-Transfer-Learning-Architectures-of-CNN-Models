# Research on Transfer Learning Architectures of CNN Models

This repository is connected via Kaggle notebooks. Updates will reflect Kaggle's version upgrades.

<sub> ****P.S.*** The project was completed under CSE498R (Research) course offered by North South University.*<sub/>

#### Abstract
This study explores the use of Convolutional Neural Networks (CNNs) with transfer learning to classify images from the SETI (Search for Extraterrestrial Intelligence) dataset, aiming to detect potential extraterrestrial signals. By leveraging advanced CNN architectures and pre-trained models across different frameworks, the research achieved significant accuracy, demonstrating the efficacy of transfer learning in specialized tasks.

<img src="https://github.com/HR-Fahim/Research-on-SETI-Data-Using-CNN-Models-with-Transfer-Learning/assets/66734379/67b0c9c4-b309-4695-8551-6ed5cdb7c3c8" alt="Flow-of-the-Transfer-Learning-in-Convolutional-Neural-Network" style="display: block; margin: 0 auto; width: 600px; height: 600px; object-fit: contain;">

#### Introduction
The SETI dataset comprises radio signal data visualized as images, necessitating sophisticated image classification techniques to identify potential extraterrestrial signals. Convolutional Neural Networks (CNNs) have proven effective for image classification tasks. This study investigates the application of transfer learning using various pre-trained CNN models to enhance the accuracy and efficiency of the classification process.

#### Methodology

**Frameworks and Models Used**

1. **TensorFlow Models:**
   - **InceptionResNetV2:** This model combines the Inception architecture with residual connections, enhancing network depth and efficiency.
   - **ResNet50:** Known for its residual learning framework, ResNet50 helps in training very deep networks by mitigating the vanishing gradient problem.
   - **MobileNetV2:** Designed for mobile and edge devices, MobileNetV2 is efficient and lightweight, making it suitable for tasks requiring lower computational resources.
   - **Performance:** The TensorFlow models achieved an accuracy of around 87%.

2. **PyTorch Models:**
   - **First PyTorch Model (95% Accuracy):**
     - **ResNet50:** Provides strong baseline performance due to its deep architecture with residual blocks.
     - **InceptionV3:** Utilizes multiple convolutional kernels at the same layer to capture diverse features, incorporating batch normalization and auxiliary classifiers for improved performance and regularization.

   - **Second PyTorch Model (94% Accuracy):**
     - **ResNet50:** Retains its strong performance due to residual connections.
     - **InceptionV4:** An advanced version of the Inception architecture that combines residual connections to further enhance feature extraction and performance.

**Data Preprocessing Methods**

1. **Data Augmentation:**
   - **Random Resized Crop:** Randomly crops a portion of the image and resizes it to a specified size, making the model robust to different scales and viewpoints.
   - **Random Horizontal Flip:** Flips the image horizontally with a 50% probability, augmenting the dataset and improving model generalization.
   - **Normalization:** Each channel of the input image is normalized to a mean and standard deviation, speeding up convergence and improving training stability.

2. **Data Transformation:**
   - **Resize and Center Crop:** Ensures uniform input image size, crucial for batch processing in CNNs.
   - **ToTensor:** Converts images to PyTorch tensors, scaling pixel values to the [0, 1] range.

**Transfer Learning and Model Specialties**

1. **Transfer Learning:**
   - **Pretrained Models:** Leveraged models pre-trained on the ImageNet dataset, providing rich feature extraction capabilities due to the vast and diverse dataset.
   - **Fine-Tuning:** Involves freezing the initial layers to retain general features learned from ImageNet and retraining the final layers to specialize in the specific task of SETI signal classification.

2. **Benefits of Transfer Learning:**
   - **Improved Performance:** Pretrained models leverage previously learned features, leading to faster convergence and higher accuracy compared to training from scratch.
   - **Reduced Training Time:** Significantly reduces the data and time required to train deep learning models, as models start with a good baseline.
   - **Better Generalization:** Pretrained models, having been trained on a large dataset, can generalize better to new, unseen data.

**Model Architecture Details**
1. **InceptionResNetV2:**
   - **Architecture:** Combines Inception modules with residual connections, allowing very deep networks without degradation.
   - **Specialty:** Balances the width and depth of the network, capturing diverse and complex features.
     
   <img src="https://github.com/HR-Fahim/Research-on-SETI-Data-Using-CNN-Models-with-Transfer-Learning/assets/66734379/b2b9bb20-cc61-470d-96a5-618317de0abe" alt="Inception-ResNet-v2-Overall-Network-Structure-and-Module-Structure-Diagrams" style="display: block; margin: 0 auto; width: 600px; height: 600px; object-fit: contain;">

2. **ResNet50:**
   - **Architecture:** Utilizes residual blocks that help in training deeper networks by adding shortcut connections.
   - **Specialty:** Effective in preventing the vanishing gradient problem, allowing for the construction of very deep networks.
     
   <img src="https://github.com/HR-Fahim/Research-on-SETI-Data-Using-CNN-Models-with-Transfer-Learning/assets/66734379/bce324c6-01ee-410c-ba0e-21763a217cb0" alt="Block-diagram-of-Resnet-50-1-by-2-architecture" style="display: block; margin: 0 auto; width: 600px; height: 600px; object-fit: contain;">

3. **MobileNetV2:**
   - **Architecture:** Uses depthwise separable convolutions and linear bottlenecks to create a lightweight model.
   - **Specialty:** Highly efficient for mobile and edge devices, balancing accuracy and computational cost.
     
   <img src="https://github.com/HR-Fahim/Research-on-SETI-Data-Using-CNN-Models-with-Transfer-Learning/assets/66734379/cc46f663-7b70-4edc-9a82-2707872474a1" alt="The-architecture-of-the-MobileNetv2-network" style="display: block; margin: 0 auto; width: 600px; height: 300px; object-fit: contain;">

4. **InceptionV3 and InceptionV4:**
   - **Architecture:** Incorporates Inception modules applying multiple convolutional filters to the same input to capture various types of features.
   - **Specialty:** Enhances the network's ability to extract rich image features, leading to improved performance.
     
   <img src="https://github.com/HR-Fahim/Research-on-SETI-Data-Using-CNN-Models-with-Transfer-Learning/assets/66734379/af50b155-dc56-4963-bae6-3b01068cff0a" alt="The-architecture-of-Inception-V3-model" style="display: block; margin: 0 auto; width: 600px; height: 300px; object-fit: contain;">
   
   <img src="https://github.com/HR-Fahim/Research-on-SETI-Data-Using-CNN-Models-with-Transfer-Learning/assets/66734379/c9acee22-c1f9-444b-8d39-b120b75cb735" alt="The-architecture-of-inception-v4-model" style="display: block; margin: 0 auto; width: 600px; height: 600px; object-fit: contain;">

#### Results and Discussion

**TensorFlow Models:**
The TensorFlow models, including InceptionResNetV2, ResNet50, and MobileNetV2, achieved an accuracy of approximately 87%. This performance demonstrates the capability of these architectures in extracting relevant features from the SETI dataset.

**First PyTorch Model:**
Combining ResNet50 and InceptionV3, the first PyTorch model reached a high accuracy of 95%. The combination of residual connections and inception modules significantly enhanced feature extraction and classification performance.

**Second PyTorch Model:**
The second PyTorch model, using ResNet50 and InceptionV4, achieved an accuracy of 94%. The advanced inception architecture and residual connections further improved the model's performance.

#### Conclusion
The research demonstrates the power of transfer learning and advanced CNN architectures in enhancing the performance of models for specialized tasks such as SETI signal classification. By leveraging pre-trained models and fine-tuning them, significant improvements in accuracy were achieved, making this approach highly effective for the SETI dataset. The unique preprocessing techniques and careful selection of architectures further contributed to the success of this research. Future work may explore further enhancements and additional architectures to push the boundaries of SETI signal detection.
