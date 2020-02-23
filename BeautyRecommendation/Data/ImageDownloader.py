import requests
import urllib
import os
import re
from bs4 import BeautifulSoup
import Data.yolo_image as Y
import json
from io import BytesIO

from PIL import Image, ImageFont, ImageDraw
# save_path = "./foul/"

# yolo = Y.YOLO()
def getBeautyPages():
    trainingData_path = "./TrainingData/"
    basic_url = 'https://www.javbus.com/actresses/'
    url = basic_url
    urlList=[]
    pageIndex=2
    while len(urlList)<500:
        response = requests.get(url)
        html = BeautifulSoup(response.text, "html.parser")
        aList = html.find_all("a",class_="avatar-box text-center")
        for a in aList:
            beautyName = a.find("span").string
            save_path = os.path.join(trainingData_path, beautyName)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            dirs = os.listdir(save_path)
            save_count=0
            for file in dirs:
                file_path = os.path.join(save_path, file)
                if os.path.isfile(file_path):
                    save_count += 1
            if save_count<=5:
                beautyURL = a['href']
                beautyTuple = (beautyName,save_count,beautyURL)
                urlList.append(beautyTuple)
                print(beautyTuple)
            if save_count>5:
                delNum = save_count-5
                for i in range(delNum):
                    file_path = os.path.join(save_path, dirs[i])
                    os.remove(file_path)
                    print("Delete",file_path)
        url=basic_url+str(pageIndex)
        pageIndex+=1
    return urlList


def downloadPageImage(yolo,beautyName,save_count,beautyURL):
    save_path = "./TrainingData/"
    url=beautyURL
    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        return 1
    # print(response.text)

    html = BeautifulSoup(response.text, "html.parser")
    # print(soup.div['class'])


    save_path=save_path+beautyName+"/"

    save_count =save_count
    print("-------------",beautyName,"-------------")
    movieBoxList = html.find_all("a",class_="movie-box")
    unsave_count =0
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]

    for movieBox in movieBoxList:
        if save_count >= 5:
            break
        href = movieBox.div.img['src']
        print(href)
        fileName = movieBox.find("date").string
        print(fileName)
        imagePath = save_path + fileName+".jpg"
        if not os.path.exists(imagePath):
            try:
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(href, imagePath)
                image = Image.open(imagePath)
                predicted_class_list = yolo.detect_image(image)
                peopleCount=predicted_class_list.count("person")
                if peopleCount!=1:
                    os.remove(imagePath)
                else:
                    save_count += 1
            except BaseException as e:
                print("Fail")
                print(e)
        else:
            unsave_count+=1

urlList = getBeautyPages()
yolo = Y.YOLO()
for beautyName,save_count,beautyURL in urlList:
    downloadPageImage(yolo,beautyName,save_count,beautyURL)