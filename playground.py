import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import string

current_path = os.getcwd()
current_path_linux = current_path.replace("\\", "/")
class_names = ["negative", "neutral", "positive"]

review_model = tf.keras.models.load_model(f"{current_path_linux}/model_experiments/model_1_bert_embeddings_01.h5", custom_objects={"KerasLayer": hub.KerasLayer})


def review_model_predict(review):
    """
    Returns the predicted class and its probability for a given review. 
    """
    # Remove all non-alphabetical characters from the review
    review = "".join([char for char in review if char in string.ascii_letters or char == " "])

    # Split the review into words, and remove any empty words
    review = " ".join(review.split())
    review = [review]

    # Use the model to make a prediction
    prediction = review_model.predict(review)

    # Get the predicted class and its probability
    predictied_class = class_names[tf.argmax(prediction[0])]
    class_prob = np.max(prediction) * 100
    return predictied_class, class_prob


while True:    
    user_review_input = input("Enter your review: ").lower()
    predictied_class, class_prob = review_model_predict(user_review_input)

    print(f"Predicted class: {predictied_class} with probability of {round(class_prob)}%")