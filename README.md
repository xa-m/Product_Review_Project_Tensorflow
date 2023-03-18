# Product Review Classifier with Tensorflow (Transfer Learning)
This Project is a review classifier which trained with 170k product reviews which are labeled by hand. The dataset itself downloaded by [Kaggle](https://www.kaggle.com/datasets/mansithummar67/171k-product-review-with-sentiment-dataset?select=sentiment.csv) and the model used in here is a pre-trained model called [BERT](https://arxiv.org/abs/1810.04805) which is a state-of-the-art model for NLP tasks. The model is trained with Tensorflow 2.0 and Keras. It performs roughly 90% accuracy on the test set which is 10% of the dataset. The model is trained with 3 epochs and the batch size is 32. The model is trained with a GPU and it takes roughly 1 hour to train the model. 
## How to use it
The model is saved in the `model_experiments` folder which is not included because of the file size. You need to train your model again in your device but because of the `tf.random.set_seed(42)` you'll get the same results as I did. The dataset is saved in the `data` folder. The model is trained with the `Product_Review_Model.ipynb` file and the `playground.ipynb` file is where you wanna use it. There is also file called `playground.py` which is the same as the `playground.ipynb` file but it is a python file.

## Required Libraries
- Tensorflow 2.9
- **Tensorflow Text 2.9**
- Tensorflow Hub 
- Numpy
- Pandas 
- Matplotlib 

**Note:** Importing tensorflow text is important because it is a dependency for the BERT model. The syntax will be like this: `import tensorflow_text as text` because of the BERT model uses it with the `text` name.


### Contact

In order to contact with me you can email me via midaskaya@outlook.com which is my personal email. You can also contact with me via Discord which my username is `.raw#5004`.
