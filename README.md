# Quick-Draw
This project implements an image classification model using a subset of the Quick Draw dataset, focusing on 30 distinct classes. The Quick Draw dataset is a large collection of hand-drawn sketches, and the aim of this project is to classify these sketches into predefined categories.
The backbone of the model is ResNet18, a well-known convolutional neural network pre-trained on the ImageNet dataset. Using a smaller set of 30 classes allowed for more focused training and faster convergence, while still maintaining strong classification performance.
I have implemented a visualization feature that displays 5 randomly selected images from the dataset along with their corresponding predicted labels. This visualization helps in better understanding the structure of the dataset and the variety of sketches within each class, providing valuable insights into the characteristics of the data.

![download](https://github.com/user-attachments/assets/428d7ff6-58f2-4d9d-b2c3-4baaad364cef)


To further assess the model's weaknesses, I conducted an error analysis. For each misclassified image, I display the original image along with both the correct label and the predicted label. 

![download (1)](https://github.com/user-attachments/assets/94f82161-2757-475b-a3ac-bacd5b989b06)     ![download (2)](https://github.com/user-attachments/assets/fd724305-b63b-4d8a-afcc-8aa1a75fdb3c)

This helps identify patterns in the types of errors made by the model, such as confusion between similar classes.

![download (3)](https://github.com/user-attachments/assets/b22c1ac4-9eb7-42c5-ad75-be85a6513b5c)

Moreover, a confusion matrix is generated to provide a comprehensive overview of the model’s performance across all 30 classes. The confusion matrix highlights which classes are often confused with each other, offering insights into potential areas for improvement. This analysis provides a deeper understanding of the model's behavior and helps to identify areas where further tuning or augmentation may improve performance

colab link: https://colab.research.google.com/drive/1K3uySK59roQNokGht7QIwH6y3RDMzO5Q?usp=sharing
