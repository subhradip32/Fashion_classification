# Fashion detection AI

## Product Description
This is a build model for detection of fashion items from the input image from the user. I had used fashion mnist to train my model. This project is made for the submission of the teachnook major project. 

## Data 
To train our model I had used the fashion mnist dataset which comes with the tensorflow module itself. This dataset have around 10 different classes of fashion. 
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```
Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

|Split	| Examples|
|-------|---------|
|'test'	|  10,000 |
|'train'|  60,000 |

to visaualise the dataset here is an example of the dataset.![fashion_mnist-3 0 1](https://github.com/subhradip32/Fashion_classification/assets/83198378/248834d5-9d9d-4812-acc8-c8ade27705aa) 
and To understand more about the data you can visit this link (https://www.tensorflow.org/datasets/catalog/fashion_mnist).

## Model
I have designed my own model to get the optimum result to from the dataset. So I build a model architecture with 3 CNN network with 3 Maxpooling technique and then the data is beign flatten and then processed by the dense layers with activation function relu and softmax.
This TensorFlow/Keras model is designed for image classification tasks, particularly suited for datasets with images of size 28x28 pixels. The model architecture consists of a series of convolutional layers followed by max-pooling layers, ultimately flattening the output for classification through fully connected layers.
```
model = tf.keras.Sequential([
    keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    keras.layers.Conv2D(32,(3,3),activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(32,(3,3),activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(32,(3,3),activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(), 
    keras.layers.Dense(64,activation= "relu"),
    keras.layers.Dense(128,activation= "relu"),
    keras.layers.Dense(64,activation= "relu"),
    keras.layers.Dense(len(class_names) , activation="softmax")
])
```
**The architecture of the model is as follows:**
1. **Input Layer:** Reshape layer to transform input images into the shape (28, 28, 1).
2. **Convolutional Layer 1:** 32 filters of size (3,3) with ReLU activation.
3. **Max Pooling Layer 1:** Downsampling using max-pooling.
4. **Convolutional Layer 2:** 32 filters of size (3,3) with ReLU activation.
5. **Max Pooling Layer 2:** Downsampling using max-pooling.
6. **Convolutional Layer 3:** 32 filters of size (3,3) with ReLU activation.
7. **Max Pooling Layer 3:** Downsampling using max-pooling.
8. **Flatten Layer:** Flattens the output from the convolutional layers.
9. **Dense Layer 1:** 64 neurons with ReLU activation.
10. **Dense Layer 2:** 128 neurons with ReLU activation.
11. **Dense Layer 3:** 64 neurons with ReLU activation.
12. **Output Layer:** Dense layer with the number of neurons equal to the number of classes in the dataset, activated using softmax.

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 reshape_1 (Reshape)         (None, 28, 28, 1)         0         
                                                                 
 conv2d_4 (Conv2D)           (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 13, 13, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_5 (Conv2D)           (None, 11, 11, 32)        9248      
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 5, 5, 32)          0         
 g2D)                                                            
                                                                 
 conv2d_6 (Conv2D)           (None, 3, 3, 32)          9248      
                                                                 
 max_pooling2d_5 (MaxPoolin  (None, 1, 1, 32)          0         
 g2D)                                                            
                                                                 
 flatten_1 (Flatten)         (None, 32)                0         
                                                                 
 dense_4 (Dense)             (None, 64)                2112      
                                                                 
 dense_5 (Dense)             (None, 128)               8320      
                                                                 
 dense_6 (Dense)             (None, 64)                8256      
                                                                 
 dense_7 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 38154 (149.04 KB)
Trainable params: 38154 (149.04 KB)
Non-trainable params: 0 (0.00 Byte)
```
The model is compiled using the Sparse Categorical Crossentropy loss function. This loss function is suitable for multi-class classification problems where the target labels are integers (sparse). Setting from_logits=True indicates that the model's output is considered as logits, i.e., the raw predicted values, rather than probabilities obtained through a softmax activation. This allows the loss function to apply softmax internally for numerical stability.
```
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
```
And here is the accuracy graph generated by the model. 
![download](https://github.com/subhradip32/Fashion_classification/assets/83198378/c6209df3-4115-45ef-bd3a-0cae151a974d)

## Model deployment
Streamlit is a Python library for quickly creating interactive web applications with minimal code, primarily focused on data science and machine learning tasks.
The model is deployed as an interactive web application using Streamlit, a Python library for creating web applications with minimal code. Users can upload images of clothing items, and the model will classify them into one of the predefined categories. To run the Streamlit application, users can execute the following command in their terminal:
**To run a Streamlit application named "streamlit_pp.py", you can use the following command in your terminal or command prompt:**
```
streamlit run streamlit_pp.py
```
This command instructs Streamlit to execute the Python script named "streamlit_pp.py" and launch the corresponding web application.
![Screenshot from 2024-03-03 11-00-04](https://github.com/subhradip32/Fashion_classification/assets/83198378/de25bbb2-d67e-406c-bc0d-f9347bd342b8)
**Now choose the image and there will be a prediction of the model.**

## Summary
I had tried to use multiple model optimization methods to get maximum accuracy like- batch normalisation , dropout,regularization and many more. 
```
Epoch 16/16
2625/2625 [==============================] - 7s 3ms/step - loss: 0.1104 - accuracy: 0.9590 - val_loss: 0.4967 - val_accuracy: 0.8925
```
As you can see this is the best I can get in this dataset without the data-augmentation, which is around **96%**.
