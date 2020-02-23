import os
# import cv2
from PIL import Image
import numpy as np
import CNN
class BeautyRecommendation:
    def __init__(self):
        pass

    def loadTrainingData(self):
        trainingDataPath = "./Data/TrainingData/"
        # full_path = os.path.abspath(os.path.join(trainingDataPath, dir_item))
        dirs = os.listdir(trainingDataPath)
        inputDataList = []
        labelList = []
        for dir in dirs:
            dir_path = os.path.join(trainingDataPath, dir)
            if os.path.isdir(dir_path):
                imageNames = os.listdir(dir_path)
                imageDataList = []
                for imageName in imageNames:
                    img_path = os.path.join(dir_path, imageName)
                    image = Image.open(img_path)
                    image = self.letterbox_image(image, (224,224))
                    image_data = np.array(image, dtype='float32')
                    image_data /= 255.

                    imageDataList.append(image_data)
                inputDataList.append(np.array(imageDataList))

                labelList.append(dir)
        inputData = np.array(inputDataList)
        inputData = np.transpose(inputData,(1,0,2,3,4))
        # inputData = np.reshape(inputData,(1000,5,224,224,3))
        labels = np.array(labelList)
        return inputData,labels



    def letterbox_image(self,image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image
    def trainModel(self):
        traningData,labels = self.loadTrainingData()
        cnn = CNN.CNN()
        cnn.build_model()
        cnn.train(traningData,labels,batch_size=128,nb_epoch=500,data_augmentation=False)
        cnn.save_model()
br = BeautyRecommendation()
br.trainModel()

