import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from Data.yolo3.model import yolo_eval, yolo_body, tiny_yolo_body,box_iou
from Data.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.generate()
        # Initialize the parameters
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        self.inpWidth = 416  # Width of network's input image
        self.inpHeight = 416  # Height of network's input image

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        prediction = self.yolo_model.predict(image_data)
        boxes = self.postProcess(prediction,self.anchors,len(self.class_names),(image.size[1],image.size[0]))
        # print('Found {} boxes for {}'.format(len(boxes), 'img'))
        #
        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #                           size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 300
        predicted_class_list=[]
        for top,left,bottom,right,box_classes,box_class_scores in boxes:
            box_classes=int(box_classes)
            predicted_class = self.class_names[box_classes]
            predicted_class_list.append(predicted_class)
        return predicted_class_list

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    # Remove the bounding boxes with low confidence using non-maxima suppression

    def postProcess(self,feats,anchors, num_classes, image_shape):
        """Convert final layer features to bounding box parameters."""
        # Reshape to batch, height, width, num_anchors, box_params.
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        result = []
        boxes_list = []
        box_scores_list = []
        for l in range(len(feats)):
            anchor = anchors[anchor_mask[l]]
            anchors_tensor = np.reshape(anchor, [1, 1, 1, len(anchor), 2])
            feat = feats[l]
            grid_shape = np.shape(feat)[1:3] # height, width
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                            [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                            [grid_shape[0], 1, 1, 1])
            grid = np.concatenate((grid_x, grid_y),axis=-1)
            # grid = np.concatenate((grid_x, grid_y))
            feat = np.reshape(
                feat, [-1, grid_shape[0], grid_shape[1], 3, 80 + 5])


            box_confidence = self.sigmoid(feat[..., 4:5])
            box_class_probs = self.sigmoid(feat[..., 5:])

            input_shape = np.array([416,416])

            # Adjust preditions to each spatial grid point and anchor size.
            box_xy = (self.sigmoid(feat[..., :2]) + grid) / grid_shape[::-1]
            box_wh = np.exp(feat[..., 2:4]) * anchors_tensor / input_shape[::-1]


            box_yx = box_xy[..., ::-1]
            box_hw = box_wh[..., ::-1]
            new_shape = np.round(np.multiply(image_shape,min(input_shape / image_shape)))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape
            box_yx = np.multiply((box_yx - offset), scale)
            box_hw *= scale

            box_mins = box_yx - (box_hw / 2.)
            box_maxes = box_yx + (box_hw / 2.)
            boxes = np.concatenate((
                box_mins[..., 0:1],  # y_min
                box_mins[..., 1:2],  # x_min
                box_maxes[..., 0:1],  # y_max
                box_maxes[..., 1:2]  # x_max
            ),axis=-1)

            # Scale boxes back to original image shape.
            boxes = np.multiply(boxes, np.concatenate((image_shape, image_shape)))
            box_scores = np.multiply(box_confidence,box_class_probs)

            #NMS
            boxes = np.reshape(boxes, [-1, 4])
            box_scores = np.reshape(box_scores, [-1, num_classes])
            boxes_list.append(boxes)
            box_scores_list.append(box_scores)
        boxes_list = np.concatenate(boxes_list, axis=0)
        box_scores_list = np.concatenate(box_scores_list, axis=0)
        for c in range(num_classes):
            scores = box_scores_list[:,c]
            boxes_class = np.insert(boxes_list,4, c,1)
            boxes_class = np.insert(boxes_class,5, scores,1)
            mask = scores>self.confThreshold
            boxes_class = boxes_class[mask]
            if len(boxes_class)>0:
                boxes_class=sorted(boxes_class,key=lambda x:x[-1],reverse=True)
                iou_list = np.array(boxes_class)

                while len(iou_list)>0:
                    candidate_box= iou_list[0]
                    result.append(candidate_box)
                    if len(iou_list)==1:
                        break
                    iou_list = iou_list[1:]

                    b1_mins = iou_list[:,:2]
                    b2_mins = candidate_box[:2]
                    b1_maxs = iou_list[:,2:4]
                    b2_maxs = candidate_box[2:4]
                    intersect_mins = np.maximum(b1_mins, b2_mins)
                    intersect_maxes = np.minimum(b1_maxs, b2_maxs)
                    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
                    b1_area = (b1_maxs[:,0] - b1_mins[:,0]) * (b1_maxs[:,1] - b1_mins[:,1])
                    b2_area = (b2_maxs[0] - b2_mins[0]) * (b2_maxs[1] - b2_mins[1])
                    iou = intersect_area / (b1_area + b2_area - intersect_area)
                    iou_mask = iou<=self.iou
                    # iou_mask =np.concatenate(iou_mask,axis=0)
                    iou_list = iou_list[iou_mask]
            #Without nms
            # boxes_class = np.concatenate((boxes,box_scores),axis=-1)[0]
            # for i in range(len(boxes_class)):
            #     grid_y_list = boxes_class[i]
            #     for j in range(len(grid_y_list)):
            #         grid = grid_y_list[j]
            #         for k in range(len(grid)):
            #             box = grid[k]
            #             for m in range(4,len(box)):
            #                 probability = box[m]
            #                 if probability>self.confThreshold:
            #                     result.append((box[0],box[1],box[2],box[3],m-4,probability))
        return result

if __name__ == '__main__':
    img = '.\TestData\\soccer_foul.jpg'
    image = Image.open(img)
    yolo = YOLO()
    r_image = yolo.detect_image(image)
    r_image.show()
