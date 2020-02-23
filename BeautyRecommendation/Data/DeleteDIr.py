import os

trainingData_path = "./TrainingData/"
dirs = os.listdir(trainingData_path)
for dirName in dirs:
    dir_path = os.path.join(trainingData_path, dirName)
    if os.path.isdir(dir_path):
        files = os.listdir(dir_path)
        if len(files)<5:
            for i in range(len(files)):
                file_path = os.path.join(dir_path, files[i])
                os.remove(file_path)
                print("Delete file", file_path)
            os.removedirs(dir_path)
            print("Delete dir", dir_path)

        elif len(files)>5:
            delNum = len(files) - 5
            for i in range(delNum):
                file_path = os.path.join(dir_path, files[i])
                os.remove(file_path)
                print("Delete file", file_path)


