I am thrilled to share my recent adventure where I dove into the intriguing world of emotion AI 🤖. I built a machine learning model using a Convolutional Neural Network (CNN)🧠, a super powerful tool when it comes to image analysis tasks.

The dataset for the project consisted of grayscale images of human faces, neatly categorized into three emotion classes - Angry 😠, Happy 😄, and Sad 😢. Each class contained around 100 images. These images were loaded and preprocessed using OpenCV, a process that included resizing to a standard size and normalizing pixel values.

The architecture of the CNN model included two convolutional layers, each followed by a max-pooling layer. The convolutional layers extract features from the image while the max-pooling layers reduce the spatial dimensionality of the model's internal representation, helping to make the model more robust. A flatten layer was then used to convert the 2D data into a 1D vector, which was passed to two fully connected (dense) layers for final classification.

The model was compiled with the Adam optimizer and Sparse Categorical Crossentropy as the loss function. It was then trained on the preprocessed images and their corresponding labels.

The performance of the model was evaluated on a held-out test set, providing a realistic estimate of how well the model can generalize to new, unseen data. The model's predictions on this test set were compared to the actual labels, generating a classification report with precision, recall, and F1-score for each emotion class.

Finally, the trained model was used to make a prediction on a new, unseen image 🖼️. This image was loaded, preprocessed in the same way as the training images, and then passed to the model for prediction. The model's output is a probability distribution over the emotion classes, and the class with the highest probability was selected as the predicted emotion for the image.

While the current model performance is modest, it's important to note that the model was trained on a relatively small dataset of around 100 images per category. In future work, I plan to leverage larger datasets to improve the model's accuracy and robustness.

This project has been a phenomenal opportunity to delve into the field of emotion AI and computer vision, and I'm super excited about the potential applications of this technology in creating more intuitive and responsive AI systems 👀.



#AI #MachineLearning #DeepLearning #DataScience #EmotionAI #ComputerVision
