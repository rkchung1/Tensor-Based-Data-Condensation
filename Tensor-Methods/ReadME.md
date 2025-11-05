# In this folder, we will describe the pipeline and the code for the tensor methods for dataset condensation - Core Tensor Route

## 1. Downloading the dataset
We download the dataset [ could be done by manually downloading it, or in this case, we utilized a script to download FashionMNIST ]

## 2. Dataset Preparation
Depending on the data type, we must assess if we need to run a auto-encoder on the dataset first. In the case of CoverType, we must run the autoencoder to perform dimensionality matching, and to reduce noise. 

## 3. Splitting 
We use a 80/20 split on the dataset. 80% will be used for training, and 20 % for testing. Out of the 20%, a part of the testing set will be used for fine tuning as well.

## 4. Create Noisy Slices
Using the script, we will create the desired amount of noisy slices into the original dataset. The slices are just the original data points with added random Gaussian Noise. The Gaussian noise scale can be configured as well.

## 5. Partial Tucker Decomposition
Perform partial tucker, with the desired number of samples and features, and using whichever gpu is available from the server (this code sens it to the server, if none are available or there are no gpus, it will run on the cpu)

## 6. Feature Projection 
We use the features from the result of Tucker Decomposition to project onto the testing set. This is done in order for both sets to have the same amount of features (same dimension). 

## 7. MLP
We run the MLP using the core tensor as the input. The output of the MLP is the relationship between the original labels and the compressed samples.


# K - Means Route :
# In this folder, we will describe the pipeline and the code for the tensor methods for dataset condensation - Core Tensor Route

## 1. Downloading the dataset
We download the dataset [ could be done by manually downloading it, or in this case, we utilized a script to download FashionMNIST ]

## 2. Dataset Preparation
Depending on the data type, we must assess if we need to run a auto-encoder on the dataset first. In the case of CoverType, we must run the autoencoder to perform dimensionality matching, and to reduce noise. 

## 3. Splitting 
We use a 80/20 split on the dataset. 80% will be used for training, and 20 % for testing. Out of the 20%, a part of the testing set will be used for fine tuning as well.

## 4. Create Noisy Slices
Using the script, we will create the desired amount of noisy slices into the original dataset. The slices are just the original data points with added random Gaussian Noise. The Gaussian noise scale can be configured as well.

## 5. Partial Tucker Decomposition
Perform partial tucker, with the desired number of samples and features, and using whichever gpu is available from the server (this code sens it to the server, if none are available or there are no gpus, it will run on the cpu)

## 6. Feature Projection 
Project the samples using the compressed features. It will just compress the features on the original dataset, and leave the samples intact.

## 7.Run k-means 
Choose the amount of centroids (k). This method will use soft labels, meaning it will output the probability distribution for each label and class, instead of choosing the centroid as the true label for the dataset.







