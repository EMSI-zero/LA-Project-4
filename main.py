import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

FOLDER = "./Dataset/"
FILES = os.listdir(FOLDER)
TEST_DIR = "./Testset/"

def load_images_train_and_test(TEST):
    """loads images into matrix sets

    Args:
        TEST (str): file path

    Returns:
        np arrays: result matrices
    """
    test=np.asarray(Image.open(TEST)).flatten()
    train=[]
    for name in FILES:
        train.append(np.asarray(Image.open(FOLDER + name)).flatten())
    train= np.array(train)
    return test,train
   
def normalize(test,train):
    """
    Normalize test and train and return them properly
    
    Args:
        test(np array) : test set
        train(np array): train set
    
    Returns:
        np array: normalized test set
        np array: normalized train set
    """
    # return normalized_test,normalized_train
    return test - np.mean(train,axis=0) , train - np.mean(train,axis=0)

def svd_function(images):
    """
    implement SVD (using np.linalg.svd) and return u,s,v 

    Args:
        images(np array): set of images
        
    Returns:
        Singular Values
    """
    # return None,None,None
    U , S , Vt = np.linalg.svd(images, full_matrices=False)
    return U , S , Vt

def project_and_calculate_weights(img,u):
    """
    calculate element wise multiplication of img and u 
    """
    return np.multiply(img, u)

def predict(test,train):
    """
    Find the most similar face to test among train set by calculating errors and finding the face that has minimum error
    
    Args:
        test(np array) : test normalized set
        train(np array) : train normalized set
    
    Returns :
        index of the data that has minimum error in train dataset
    
    """
    maxS = 0 #index of most similar face
    minErr = np.linalg.norm(train[0,0] - test) #initiate minimun error face
    for i in range(0,len(train[0])):
        error = np.linalg.norm(train[:,i] - test)
        if(minErr >= error):
            minErr = error
            maxS = i
    return maxS
    

def plot_face(tested,predicted):
    """
    Plot tested image and predicted image . 

    """
    figure = plt.figure()
    ax = figure.add_subplot(1,2,1)
    plt.imshow(tested, cmap="gray")
    ax.set_title("tested")
    ax = figure.add_subplot(1,2,2)
    plt.imshow(predicted, cmap="gray")
    ax.set_title("predicted")
    plt.show()

if __name__ == "__main__":
    true_predicts=0
    all_predicts=0
    for TEST_FILE in os.listdir(TEST_DIR):
        # Loading train and test
        test,train=load_images_train_and_test(TEST_DIR+TEST_FILE)

        # Normalizing train and test
        test,train=normalize(test,train)
        test=test.T
        train=train.T
        test = np.reshape(test, (test.size, 1))

        # Singular value decomposition
        u,s,v=svd_function(train)

        # Weigth for test
        w_test=project_and_calculate_weights(test,u)
        w_test=np.array(w_test, dtype='int8').flatten()

        # Weights for train set
        w_train=[]
        for i in range(train.shape[1]):
            w_i=project_and_calculate_weights(np.reshape(train[:, i], (train[:, i].size, 1)),u)
            w_i=np.array(w_i, dtype='int8').flatten()
            w_train.append(w_i)
        w_train=np.array(w_train).T

        # Predict 
        index_of_most_similar_face=predict(w_test,w_train)

        # Showing results
        print("Test : "+TEST_FILE)
        print(f"The predicted face is: {FILES[index_of_most_similar_face]}")
        print("\n***************************\n")

        # Calculating Accuracy
        all_predicts+=1
        if FILES[index_of_most_similar_face].split("-")[0]==TEST_FILE.split("-")[0]:
            true_predicts+=1
            # Plotting correct predictions 
            plot_face(Image.open(TEST_DIR+TEST_FILE),Image.open(FOLDER+FILES[index_of_most_similar_face]))
        else:
            # Plotting wrong predictions
            plot_face(Image.open(TEST_DIR+TEST_FILE),Image.open(FOLDER+FILES[index_of_most_similar_face]))

    # Showing Accuracy
    accuracy=true_predicts/all_predicts
    print(f'Accuracy : {"{:.2f}".format(accuracy*100)} %')
        
    