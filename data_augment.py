# Copyright 2016 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================

"""Provides utilities to preprocess images for the Inception networks."""



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import tensorflow as tf

import random

import numpy as np

import os,cv2

from PIL import Image, ImageEnhance

from tensorflow.python.ops import control_flow_ops





def apply_with_random_selector(x, func, num_cases):

    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:

    x: input Tensor.

    func: Python function to apply.

    num_cases: Python int32, number of cases to sample sel from.

    Returns:

    The result of func(x, sel), where func receives the value of the

    selector as a python integer, but sel is sampled dynamically.

    """

    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)

    # Pass the real x only to one of the func calls.

    return control_flow_ops.merge([

        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)

        for case in range(num_cases)])[0]





def distort_color(image, color_ordering=0, fast_mode=True, scope=None):

    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops

    matters. Ideally we would randomly permute the ordering of the color ops.

    Rather then adding that level of complication, we select a distinct ordering

    of color ops for each preprocessing thread.

    Args:

    image: 3-D Tensor containing single image in [0, 1].

    color_ordering: Python int, a type of distortion (valid values: 0-3).

    fast_mode: Avoids slower ops (random_hue and random_contrast)

    scope: Optional scope for name_scope.

    Returns:

    3-D Tensor color-distorted image on range [0, 1]

    Raises:

    ValueError: if color_ordering not in [0, 3]

    """

    with tf.name_scope(scope, 'distort_color', [image]):

        # 修改－调整参数

        lower, upper = .75, 1.25

        max_delta = .15

        hue_delta = .15

        if fast_mode:

            if color_ordering == 0:

                image = tf.image.random_brightness(image, max_delta=max_delta)

                image = tf.image.random_saturation(image, lower=lower, upper=upper)

            else:

                image = tf.image.random_saturation(image, lower=lower, upper=upper)

                image = tf.image.random_brightness(image, max_delta=max_delta)

        else:

            if color_ordering == 0:

                image = tf.image.random_brightness(image, max_delta=max_delta)

                image = tf.image.random_saturation(image, lower=lower, upper=upper)

                image = tf.image.random_hue(image, max_delta=hue_delta)

                image = tf.image.random_contrast(image, lower=lower, upper=upper)

            elif color_ordering == 1:

                image = tf.image.random_saturation(image, lower=lower, upper=upper)

                image = tf.image.random_brightness(image, max_delta=max_delta)

                image = tf.image.random_contrast(image, lower=lower, upper=upper)

                image = tf.image.random_hue(image, max_delta=hue_delta)

            elif color_ordering == 2:

                image = tf.image.random_contrast(image, lower=lower, upper=upper)

                image = tf.image.random_hue(image, max_delta=hue_delta)

                image = tf.image.random_brightness(image, max_delta=max_delta)

                image = tf.image.random_saturation(image, lower=lower, upper=upper)

            elif color_ordering == 3:

                image = tf.image.random_hue(image, max_delta=hue_delta)

                image = tf.image.random_saturation(image, lower=lower, upper=upper)

                image = tf.image.random_contrast(image, lower=lower, upper=upper)

                image = tf.image.random_brightness(image, max_delta=max_delta)

            else:

                raise ValueError('color_ordering must be in [0, 3]')

    

    # The random_* ops do not necessarily clamp.

    return tf.clip_by_value(image, 0.0, 1.0)





def distorted_bounding_box_crop(image,

                                bbox,

                                min_object_covered=0.1,

                                aspect_ratio_range=(0.75, 1.33),

                                area_range=(0.6, 1.0),

                                max_attempts=100,

                                scope=None):

    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:

        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).

        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]

        where each coordinate is [0, 1) and the coordinates are arranged

        as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole

        image.

        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped

        area of the image must contain at least this fraction of any bounding box

        supplied.

        aspect_ratio_range: An optional list of `floats`. The cropped area of the

        image must have an aspect ratio = width / height within this range.

        area_range: An optional list of `floats`. The cropped area of the image

        must contain a fraction of the supplied image within in this range.

        max_attempts: An optional `int`. Number of attempts at generating a cropped

        region of the image of the specified constraints. After `max_attempts`

        failures, return the entire image.

        scope: Optional scope for name_scope.

    Returns:

        A tuple, a 3-D Tensor cropped_image and the distorted bbox

    """

    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):

        # Each bounding box has shape [1, num_boxes, box coords] and

        # the coordinates are ordered [ymin, xmin, ymax, xmax].

    

        # A large fraction of image datasets contain a human-annotated bounding

        # box delineating the region of the image containing the object of interest.

        # We choose to create a new bounding box for the object which is a randomly

        # distorted version of the human-annotated bounding box that obeys an

        # allowed range of aspect ratios, sizes and overlap with the human-annotated

        # bounding box. If no box is supplied, then we assume the bounding box is

        # the entire image.

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(

            tf.shape(image),

            bounding_boxes=bbox,

            min_object_covered=min_object_covered,

            aspect_ratio_range=aspect_ratio_range,

            area_range=area_range,

            max_attempts=max_attempts,

            use_image_if_no_bounding_boxes=True)

        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box



        # Crop the image to the specified bounding box.

        cropped_image = tf.slice(image, bbox_begin, bbox_size)

    return cropped_image, distort_bbox





def preprocess_for_train(image, height, width, bbox,

                         fast_mode=True,

                         scope=None,

                         add_image_summaries=True):

    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data

    set during training in order to make the network invariant to aspects

    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different

    transformations applied to the image.

    Args:

        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be

        [0, 1], otherwise it would converted to tf.float32 assuming that the range

        is [0, MAX], where MAX is largest positive representable number for

        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).

        height: integer

        width: integer

        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]

        where each coordinate is [0, 1) and the coordinates are arranged

        as [ymin, xmin, ymax, xmax].

        fast_mode: Optional boolean, if True avoids slower transformations (i.e.

        bi-cubic resizing, random_hue or random_contrast).

        scope: Optional scope for name_scope.

        add_image_summaries: Enable image summaries.

      Returns:

        3-D float Tensor of distorted image used for training with range [-1, 1].

    """

    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):

        if bbox is None:

            bbox = tf.constant([0.0, 0.0, 1.0, 1.0],

                             dtype=tf.float32,

                             shape=[1, 1, 4])

        #if image.dtype != tf.float32:

        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Each bounding box has shape [1, num_boxes, box coords] and

        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),

                                                      bbox)

        if add_image_summaries:

            tf.summary.image('image_with_bounding_boxes', image_with_box)

     

        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)

        # Restore the shape since the dynamic slice based upon the bbox_size loses

        # the third dimension.

        distorted_image.set_shape([None, None, 3])

        image_with_distorted_box = tf.image.draw_bounding_boxes(

            tf.expand_dims(image, 0), distorted_bbox)

        if add_image_summaries:

            tf.summary.image('images_with_distorted_bounding_box',

                           image_with_distorted_box)

    

        # This resizing operation may distort the images because the aspect

        # ratio is not respected. We select a resize method in a round robin

        # fashion based on the thread number.

        # Note that ResizeMethod contains 4 enumerated resizing methods.

    

        # We select only 1 case for fast_mode bilinear.

        num_resize_cases = 1 if fast_mode else 4

        distorted_image = apply_with_random_selector(

            distorted_image,

            lambda x, method: tf.image.resize_images(x, [height, width], method),

            num_cases=num_resize_cases)

    

        if add_image_summaries:

            tf.summary.image('cropped_resized_image',

                             tf.expand_dims(distorted_image, 0))

    

        # Randomly flip the image horizontally.

        distorted_image = tf.image.random_flip_left_right(distorted_image)

    

        # Randomly distort the colors. There are 1 or 4 ways to do it.

        num_distort_cases = 1 if fast_mode else 4

        distorted_image = apply_with_random_selector(

            distorted_image,

            lambda x, ordering: distort_color(x, ordering, fast_mode),

            num_cases=num_distort_cases)

    

        if add_image_summaries:

            tf.summary.image('final_distorted_image',

                             tf.expand_dims(distorted_image, 0))

        # 修改－取消调整

        # distorted_image = tf.subtract(distorted_image, 0.5)

        # distorted_image = tf.multiply(distorted_image, 2.0)

    return distorted_image





def preprocess_for_eval(image, height, width,

                        central_fraction=0.875, scope=None):

    """Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by

    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the

    input image.

    Args:

        image: 3-D Tensor of image. If dtype is tf.float32 then the range should be

        [0, 1], otherwise it would converted to tf.float32 assuming that the range

        is [0, MAX], where MAX is largest positive representable number for

        int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).

        height: integer

        width: integer

        central_fraction: Optional Float, fraction of the image to crop.

        scope: Optional scope for name_scope.

    Returns:

        3-D float Tensor of prepared image.

      """

    with tf.name_scope(scope, 'eval_image', [image, height, width]):

        if image.dtype != tf.float32:

            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            # Crop the central region of the image with an area containing 87.5% of

            # the original image.

        if central_fraction:

            image = tf.image.central_crop(image, central_fraction=central_fraction)

    

        if height and width:

            # Resize the image to the specified height and width.

            image = tf.expand_dims(image, 0)

            image = tf.image.resize_bilinear(image, [height, width],

                                               align_corners=False)

        image = tf.squeeze(image, [0])

        image = tf.subtract(image, 0.5)

        image = tf.multiply(image, 2.0)

    return image





def preprocess_image(image, height, width,

                     is_training=False,

                     bbox=None,

                     fast_mode=True,

                     add_image_summaries=True):

    """Pre-process one image for training or evaluation.

    Args:

        image: 3-D Tensor [height, width, channels] with the image. If dtype is

          tf.float32 then the range should be [0, 1], otherwise it would converted

          to tf.float32 assuming that the range is [0, MAX], where MAX is largest

          positive representable number for int(8/16/32) data type (see

          `tf.image.convert_image_dtype` for details).

        height: integer, image expected height.

        width: integer, image expected width.

        is_training: Boolean. If true it would transform an image for train,

          otherwise it would transform it for evaluation.

        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]

          where each coordinate is [0, 1) and the coordinates are arranged as

          [ymin, xmin, ymax, xmax].

        fast_mode: Optional boolean, if True avoids slower transformations.

        add_image_summaries: Enable image summaries.

    Returns:

        3-D float Tensor containing an appropriately scaled image

    Raises:

        ValueError: if user does not provide bounding box

    """

    if is_training:

        return preprocess_for_train(image, height, width, bbox, fast_mode,

                                add_image_summaries=add_image_summaries)

    else:

        return preprocess_for_eval(image, height, width)





class DataAugmentation():

    def __init__(self):

        pass



    @staticmethod

    def openImage(image):

        return Image.open(image, mode="r")



    @staticmethod

    def randomRotation(image, mode=Image.BICUBIC):

        """

        :param mode:

        :param image:

        :return:

        """

        random_angle = np.random.randint(-10, 10)

        return image.rotate(random_angle, mode)



    @staticmethod

    def randomCrop(image):

        """

        :param image: 

        :return: 

        """

        image_width = image.size[0]

        image_height = image.size[1]

        crop_win_size = np.random.randint(40, 68)

        random_region = (

            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,

            (image_height + crop_win_size) >> 1)

        return image.crop(random_region)



    @staticmethod

    def randomColor(image):

        """

        :param image: 

        :return:

        """

        random_factor = np.random.randint(0, 17) / 10.  # 随机因子

        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度

        random_factor = np.random.randint(10, 17) / 10.  # 随机因子

        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度

        random_factor = np.random.randint(10, 17) / 10.  # 随机因1子

        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

        random_factor = np.random.randint(0, 17) / 10.  # 随机因子

        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度



    @staticmethod

    def randomGaussian(image, mean=0.4, sigma=0.4):

        """

        :param image:

        :return:

        """

        def gaussianNoisy(im, mean = mean, sigma = sigma):

            """

            :param im:

            :param mean: 

            :param sigma: 

            :return:

            """

            for _i in range(len(im)):

                im[_i] += random.gauss(mean, sigma)

            return im

        img = np.asarray(image)

        img.flags.writeable = True

        

        width, height = img.shape[:2]

        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)

        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)

        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)

        img[:, :, 0] = img_r.reshape([width, height])

        img[:, :, 1] = img_g.reshape([width, height])

        img[:, :, 2] = img_b.reshape([width, height])

        return Image.fromarray(np.uint8(img))



    @staticmethod

    def saveImage(image, path):

        image.save(path)



def makeDir(path):

    try:

        if not os.path.exists(path):

            if not os.path.isfile(path):

                os.makedirs(path)

            return 0

        else:

            return 1

    except Exception as e:

        print(str(e))

        return -2



def imageOps(func_name, image, des_path, file_name, times=2):

    funcMap = {
               "randomRotation": DataAugmentation.randomRotation,

               "randomCrop": DataAugmentation.randomCrop,

               "randomColor": DataAugmentation.randomColor,

               "randomGaussian": DataAugmentation.randomGaussian

               }

    if funcMap.get(func_name) is None:

        return -1



    for _i in range(0, times, 1):

        new_image = funcMap[func_name](image)

        print('save file name',os.path.join(des_path, file_name.split('.')[0]+'_'+func_name + str(_i) +'.'+ file_name.split('.')[1]))
        DataAugmentation.saveImage(new_image, os.path.join(des_path, file_name.split('.')[0]+'_'+func_name + str(_i) +'.'+ file_name.split('.')[1]))





opsList = {"randomRotation", "randomColor", "randomGaussian"}
# opsList = {"randomGaussian"}





def dataAug(path, new_path):

    """

    :param src_path:

    :param des_path:

    :return:

    """

    makeDir(new_path)

        

    if os.path.isdir(path):

        img_names = os.listdir(path)

    else:

        print('path must be a folder')

        return

    for img_name in img_names:

        print(img_name)

        tmp_img_name = os.path.join(path, img_name)

        if os.path.isdir(tmp_img_name):

            if makeDir(os.path.join(new_path, img_name)) != -1:

                dataAug(tmp_img_name, os.path.join(new_path, img_name))

            else:

                print('create new dir failure')

                return -1

                # os.removedirs(tmp_img_name)

        elif tmp_img_name.split('.')[1] != "DS_Store":

            image = DataAugmentation.openImage(tmp_img_name)

            for ops_name in opsList:

                imageOps(ops_name, image, new_path, img_name)



if __name__ == "__main__":
    """
    with tf.Session() as sess:

        for img_path in os.listdir('./attendees'):

            image = cv2.imdecode(np.fromfile(os.path.join('./attendees',img_path), dtype=np.uint8),-1)

            if image is None:

                continue

            height, width = image.shape[0],image.shape[1]

            distorted_image = sess.run(preprocess_image(image, height, width,is_training=True,bbox=None,fast_mode=None,add_image_summaries=True))

            M = cv2.getRotationMatrix2D((height/2,width/2),np.random.randint(-10, 10),1)

            distorted_image = cv2.warpAffine(distorted_image,M,(height, width))

#             cv2.imshow('',distorted_image)

#             cv2.waitKey(0)

            # 修改－支持中文文件名
            # print(os.path.join('./dataAug/','%s_'%np.random.randint(1000)+img_path))
            cv2.imencode('.jpg', distorted_image * 255)[1].tofile(os.path.join('./dataAug/','%s_'%np.random.randint(1000)+img_path))

            # print(os.path.join('./dataAug/','%s_'%np.random.randint(1000)+img_path))
            # cv2.imwrite(os.path.join('./dataAug/','%s_'%np.random.randint(1000)+img_path),distorted_image * 255)
    """
    dataAug('./attendees','./dataaugs/')
