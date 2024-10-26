# Digit and Greek Character Recognition with CNNs

## Project Overview

This project implements a convolutional neural network (CNN) using PyTorch for handwritten digit recognition with an extension to recognize Greek letters. Key goals include training a CNN on the MNIST dataset, exploring network feature visualizations, adapting the model for Greek character recognition via transfer learning, and deploying the model for real-time digit recognition through a live video feed. This project demonstrates various machine learning skills, including data preprocessing, model architecture design, hyperparameter optimization, and real-time inference using OpenCV.

## Technical Details

The project begins with data preparation, leveraging PyTorch’s `torchvision` to load and preprocess the MNIST dataset. For the initial model, a CNN was constructed with two convolutional layers, dropout regularization, and fully connected layers, ending in a softmax activation for classification. The training and evaluation pipeline iteratively updates the model over several epochs, tracking training and test loss, and saving the trained model to a file for reuse.

To assess the model’s learned features, convolutional filters were extracted and visualized. Applying these filters to sample images provides insight into how the model perceives edges, shapes, and textures, aiding interpretability.

An extension on transfer learning was implemented to adapt the trained CNN to Greek character recognition. Custom transformations resized and normalized the Greek letter images, aligning them with the MNIST format. The model was fine-tuned on the new dataset, allowing it to generalize well to Greek characters.

Further exploration included optimizing CNN hyperparameters, such as the number of convolutional layers, filter sizes, and dropout rates. A systematic search through the hyperparameter space led to refined configurations that reduced overfitting and maximized validation accuracy. The analysis also visualized training and validation trends, comparing accuracy across configurations.

Finally, the project demonstrated real-time deployment using OpenCV and PyTorch. A live digit recognition system was created, capturing frames from a camera, preprocessing them, and feeding them through the trained model to display predicted digits on the screen.

## Skills and Knowledge Demonstrated

- **Deep Learning Fundamentals**: Demonstrated expertise in CNN design, training, and evaluation, with knowledge of layer configurations, activation functions, and regularization methods.
- **Transfer Learning**: Successfully adapted a pre-trained model to a new character recognition task, understanding the transfer of learned features across related datasets.
- **Real-Time Computer Vision**: Implemented real-time digit classification using OpenCV and PyTorch, creating a practical application for live video feeds.
- **Data Preprocessing and Augmentation**: Applied data transformations to prepare inputs for neural networks, ensuring compatibility and optimal performance across datasets.
- **Hyperparameter Tuning and Model Optimization**: Explored various CNN configurations, systematically evaluating each to balance accuracy and computational efficiency.
- **Feature Visualization**: Gained insights into model interpretability by visualizing learned filters and applying them to input images to assess feature extraction.


