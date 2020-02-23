
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# CNN网络模型类
class CNN:
    def __init__(self):
        self.model = None

    # 建立模型
    def build_model(self):
        # base_model = ResNet50(weights='imagenet')

        # 首先，定义视觉模型
        # digit_input = base_model.output
        digit_input = Input(shape=(224,224,3))
        x = Conv2D(64, (3, 3))(digit_input)
        x = Conv2D(64, (3, 3))(x)
        x = Conv2D(64, (3, 3))(x)
        x = Conv2D(64, (3, 3))(x)
        x = Conv2D(64, (3, 3))(x)
        x = Conv2D(64, (3, 3))(x)
        x = MaxPooling2D((2, 2))(x)
        out = Flatten()(x)

        # vision_model = Model(base_model.input, out)
        vision_model = Model(digit_input, out)

        # 然后，定义区分数字的模型
        digit_a = Input(shape=(224,224,3))
        digit_b = Input(shape=(224,224,3))
        digit_c = Input(shape=(224,224,3))
        digit_d = Input(shape=(224,224,3))
        digit_e = Input(shape=(224,224,3))

        # 视觉模型将被共享，包括权重和其他所有
        out_a = vision_model(digit_a)
        out_b = vision_model(digit_b)
        out_c = vision_model(digit_c)
        out_d = vision_model(digit_d)
        out_e = vision_model(digit_e)

        concatenated = keras.layers.concatenate([out_a, out_b,out_c,out_d,out_e])
        out = Dense(800, activation='sigmoid')(concatenated)

        self.model = Model([digit_a, digit_b, digit_c , digit_d, digit_e], out)

        # 输出模型概况
        self.model.summary()
        # return model
    # 训练模型
    def train(self, trainingData,labels, batch_size=20, nb_epoch=100, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成实际的模型配置工作

        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit([trainingData[0],trainingData[1],trainingData[2],trainingData[3],trainingData[4]],
                           labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_split=0.2,
                           shuffle=True,verbose=1,)
        # 使用实时数据提升
        else:
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=False,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(trainingData)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(trainingData, labels,
                                                  batch_size=batch_size),
                                     samples_per_epoch=trainingData.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_split=0.2)

    MODEL_PATH = './model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print(f'{self.model.metrics_names[1]}:{score[1] * 100}%')
        return score

