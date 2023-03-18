import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

current_path = os.getcwd()
current_path_linux = current_path.replace("\\", "/")
class_names = ["negative", "neutral", "positive"]

review_model = tf.keras.models.load_model(f"{current_path_linux}/model_experiments/model_1_bert_embeddings_01.h5", custom_objects={"KerasLayer": hub.KerasLayer})


user_review_input = input("Enter your review: ").lower()
user_review_input = [user_review_input]

prediction = review_model.predict(user_review_input)
predictied_class = class_names[tf.argmax(prediction[0])]
print(f"Predicted class: {predictied_class}")