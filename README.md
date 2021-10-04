# MOSIBIT-Machine Learning
Machine Learning Development For Mobile SIBI Translation

## Documentation
The work of this application is divided into three steps so that it can run as we made it:
- Machine Learning Training and Prediction Modeling.
- Machine Learning Deployment in the Cloud and Testing the API.
- Kotlin Android Application Creation.

### Machine Learning Training and Prediction Model.
#### Requirement
For our training model, we were using:
- Python 3.8.6
- tensorflow 2.2.0
- numpy 1.20.3
- pandas 1.2.4
- mediapipe 0.8.3.1
- opencv-python 4.5.1
- jupyterlab 3.0.12

We suggest to using virtual environment to contain the needed module in case there's problem to them.

#### The Training Model
Before we start to start the training. First we must do the preprocessing by turning the SIBI sign language dataset into features that can we use and put them into csv file so we can start easily to train them. This also aplicable to the validation file for checking the accuracy of the model we have created.

![Plan A Preprocessing](https://user-images.githubusercontent.com/16248869/120893410-697c7180-c63d-11eb-9a2d-1e4c9093487d.jpg)

You can run this preprocessing by run these python jupyter file in a machine that installed with python and jupyter notebook in terminal or Visual Code: 
- feature_extraction_to_csv.ipynb >>> this will have two csv output called hands_SIBI_training.csv and output called hands_SIBI_validation.csv 

After running those python file, now we can proceed to start the training model. The training model we created is using method called Convolutional Neural Network. Usually the CNN is powerful to training in image file but in this case we using it as Human Activity Recognition since the features we use contain the dataset about human activity. By using Convolutional Neural Network one Dimension we had a high accuracy in recognition the sign language's alphabets.

![PLAN A Training with CNN or any comparable CNN model](https://user-images.githubusercontent.com/16248869/120893737-012e8f80-c63f-11eb-9612-cff3df8097f0.jpg)

You can  run this training and save the model by python jupyter file in a machine that installed with python and jupyter notebook in terminal or Visual Code:
- training_model.ipynb >>> this will have a h5 output called model_SIBI.h5

After we running the preprocessing and the training lastly have the h5 file model. We can proceed to the testing the model by using video stream through webcam. This also for checking is our model/system is good enough run real-time and we have tested it running in Native Android Application. This is will extract the features and prediction per frame that our machine capable do. 

You can  run this video stream and prediction model by run this python jupyter file in a machine that installed with python and jupyter notebook in terminal or Visual Code:
- predictiion_model.ipynb >>> this will show output camera webcam and the prediction of the sign language.

![unknown](https://user-images.githubusercontent.com/16248869/120894057-ad24aa80-c640-11eb-8253-63a694904345.png)
