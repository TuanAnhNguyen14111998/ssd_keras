# Nhap cac thu vien can thiet
from __future__ import division
import numpy as np
# Nhap class tao model trong Keras
from keras.models import Model
# Nhap cac class tao cac layer trong Keras
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
# Nhap class tao chuan hoa L2
from keras.regularizers import l2
# Nhap backend cua Keras
import keras.backend as K

# Nhap cac class tao cac layers custom tu Keras

# Nhap class dinh nghia cac Anchor Box
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# Nhap class dinh nghia chuan hoa L2
from keras_layers.keras_layer_L2Normalization import L2Normalization
# Nhap class cho phep viec decode cac raw tensor output thanh cac toa do bbx tuyet doi
# duoc su dung cho qua trinh predict sau khi da co model
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

# function cho phep dinh nghia model SSD300
def ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    '''
    #############################################################################
    # Build model Keras voi kien truc SSD300, xem paper de hieu ve kien truc
    #############################################################################

    Base network la mang VGG-16 duoc luoc bo cac layer classification, va model duoc mo
    rong bang kien truc SSD, nhu duoc mo ta trong paper.

    Hau het cac doi so ma function nay can duoc su dung cho viec cau hinh cac anchor
    box layers.
    
    Trong truong hop ban thuc hien trainning mang, cac tham so duoc truyen o day
    phai giong voi cac tham so duoc su dung de thiet lap `SSDBoxEncoder`.
    
    Trong truong hop ban tai trong so da duoc dao tao truoc, cac tham so duoc truyen vao o day
    phai giong voi cac tham so da tao ra duoc cac trong so duoc dao tao truoc do.

    Mot so trong so cac doi so nay se duoc giai thich chi tiet hon trong tai lieu cua class `SSDBoxEncoder`

    Luu y: Yeu cau Keras tu v2.0 tro len, va hien tai chi hoat dong voi backend
    Tensorflow (v1.0) tro len.

    ############################################################################################
    # Giai thich chi tiet cac doi so duoc truyen vao function
    ###########################################################################################

    Cac doi so:
        * image_size (tuple): 
                - Kich thuoc hinh anh dau vao voi dinh dang `(height, width, channels)`.
        * n_classes (int):
                - So luong cac class positives (cac class object, ko phai class background)
                  vi du: 20 cho Pascal VOC, 80 cho MS COCO.
        * mode (str, optional): 
                - Mot trong cac gia tri 'training', 'inference' va 'inference_fast'.
                - Trong mode 'trainning', output cua model se la raw predict tensor, trong khi do,
                  trong mode 'inference' va 'inference_fast' cac raw predict tensor se duoc decoded
                  (giai ma) thanh cac gia tri toa do tuyet doi, va duoc loc qua cac nguong tin cay
                  (confidence thresholding), non-maximum suppression, va top-k filtering.
                - Su khac biet giua hai mode 'inference' va 'inference_fast' do la 'inference' tuan theo
                  quy trinh theo trien khai Caffe ban dau, trong khi do mode 'inference_fast' su dung
                  thu tuc decoded cac prediction nhanh hon.
        * l2_regularization (float, optional):
                - Ty le chuan hoa L2. Ap dung cho tat ca cac layer convolutional.
                  Dat thanh 0 de huy bo viec su dung chuan hoa L2.
        * min_scale (float, optional): 
                - He so ty le nho nhat cho cac anchor box va no thuoc layer predict thap nhat.
        * max_scale (float, optional):
                - He so ty le lon nhat cho cac anchor box va no thuoc layer predict cao nhat. 
                - Tat ca cac he so ty le cho cac anchor box cua cac layer predict o giua hai lop predict
                  cao nhat va thap nhat se duoc noi suy tuyen tinh trong doan tu [min_scale, max_scale]
                - Ghi nho rang, he so ty le duoc noi suy tuyen tinh tu thu 2 den cuoi cung se thuc su
                  su la he so ty le cho layer predict cuoi cung, trong khi he so ty le cuoi cung se duoc
                  su dung cho second box cua aspect ratios = 1 trong layer predict cuoi cung, neu
                  `two_boxes_for_ar1` duoc dat la `True`.
        * scales (list, optional):
                - Mot list cac phan tu type la float chua cac he so ty le duoc ap dung cho tung
                  cac convolutional predictor layer. 
                - List nay phai dai hon mot phan tu so voi so luong cac predict layers. 
                - k phan tu dau tien la k cac he so ty le cho k layer predictor dau tien,
                  trong khi phan tu cuoi cung duoc su dung cho second box co aspect ratio = 1
                  trong layer predictor cuoi cung neu `two_boxes_for_ar1` duoc dat la true.
                - He so ty le bo sung nay phai thoa man hai dieu kien sau, ke ca trong
                  truong hop no khong duoc su dung: 
                    + Neu list nay duoc truyen vao, thi no se ghi de len cac gia tri 
                      min_scale va max_scale.
                    + Tat ca cac phan tu he so ty le phai lon hon 0.
        * aspect_ratios_global (list, optional):
                - Mot list chua cac aspect ratio (ty le khung hinh) se duoc dung de tao ra cac
                  anchor box.
                - List nay se duoc su dung cho toan bo cac layer trong model.
        * aspect_ratios_per_layer (list, optional):
                - Mot list chua aspect ratio cho tung lop predictor layers.
                - Dieu nay cho phep ban dat ty le khung hinh (aspect ratio) cho tung layer predictor
                  rieng le, day la truong hop trien khai cua SSD300 goc.
                - Neu list nay duoc truyen vao thi no se ghi de len `aspect_ratios_global`.
        * two_boxes_for_ar1 (bool, optional):
                - Chi lien quan den gia tri aspect ratio = 1.
                - Se bi bo qua neu la truong hop khac
                - Neu mang gia tri `True`, hai anchor box se duoc tao cho ty le khung hinh aspect
                  ratio = 1.
                  + Anchor box dau tien se duoc tao bang viec su dung scale cua lop tuong ung
                  + Anchor box thu hai se duoc tao bang viec su dung gia tri trung binh hinh hoc cua 
                    he so ty le scale cua layer dang xet va he so ty le cua layer tiep theo.
        * steps (list, optional):
                - `None` hoac la mot list chua so luong phan tu bang voi so luong cua cac layer 
                  predict.
                - Cac phan tu co the la ints/floats hoac tuples cua ints/floats. 
                - Nhung con so nay dai dien cho moi predict layer co bao nhieu pixels cach cac center
                  cua cac anchor box theo chieu doc va chieu ngang.
                - Neu list chua ca ints/floats thi gia tri do se duoc su dung cho ca hai kich thuoc
                  khong gian.
                - Neu list chua tuples cua hai ints/floats thi chung dai dien cho 
                  `(step_height, step_width)`.
                - Neu step khong duoc cung cap gia tri, thi chung se duoc tinh toan sao cho 
                  cac center point cua anchor box se tao thanh mot luoi cach deu chieu rong va chieu cao
                  cua hinh anh.
        * offsets (list, optional):
                - `None` hoac mot list chua so luong phan tu bang voi so luong cac layer predictor
                - Cac phan tu co the la floats hoac tuples cua hai floats.
                - Nhung con so nay dai dien cho moi layer predictor co bao nhieu pixels tu phia
                  tren cung ben trai cua hinh anh den cac diem trung tam (center point).
                - Co mot vai dieu quan trong o day: 
                    + The offsets are not absolute pixel values, but fractions 
                      of the step size specified in the `steps` argument.
                      If the list contains floats, then that value will
                      be used for both spatial dimensions.
                      If the list contains tuples of two floats, then they represent
                      `(vertical_offset, horizontal_offset)`. If no offsets are provided, 
                      then they will default to 0.5 of the step size.
        * clip_boxes (bool, optional):
                - Neu `true`, cat cac toa do anchor box de anchor box nam trong ranh gioi cua hinh anh.
        * variances (list, optional):
                - Mot list co 4 so floats > 0. 
                - The anchor box offset for each coordinate will be divided by
                  its respective variance value.
        * coords (str, optional):
                - Định dạng tọa độ hộp được sử dụng trong mô hình
                  tuc la day khog phai la dinh dang dau vao cua ground truth boxes.
                  Co the centroid cho dinh dang: `cx, cy, h, w`, `min, max` cho dinh dang
                  (xmin, xmax, ymin, ymax) hoac 'conners' cho dinh dang `xmin, ymin, xmax, ymax`
        * normalize_coords (bool, optional):
                - Dat thanh 'True' neu model su dung toa do tuong doi thay vi toa do tuyet doi
                  tuc la neu model du doan toa do cua cac box trong [0, 1] thay vi la toa do tuyet doi.
        * subtract_mean (array-like, optional):
                - `None` hoac mot doi tuong array chua cac so nguyen hoac cac gia tri
                  dau phay dong. 
                - The elements of this array will be subtracted from the image pixel intensity values. 
                  For example, pass a list of three integers to perform per-channel mean normalization 
                  for color images.
        * divide_by_stddev (array-like, optional): 
                - `None` or an array-like object of non-zero integers or floating point 
                  values of any shape that is broadcast-compatible with the image shape.
                  The image pixel intensity values will be divided by the elements of this array. 
                  For example, pass a list of three integers to perform per-channel standard 
                  deviation normalization for color images.
        * swap_channels (list, optional): 
                - Either `False` or a list of integers representing the desired order 
                  in which the input image channels should be swapped.
        * confidence_thresh (float, optional): 
                - A float in [0,1), the minimum classification confidence in a specific positive class 
                  in order to be considered for the non-maximum suppression stage for the respective class.
                - A lower value will result in a larger part of the selection process being 
                  done by the non-maximum suppression stage, while a larger value will result in a 
                  larger part of the selection process happening in the confidence thresholding stage.
        * iou_threshold (float, optional): 
                - A float in [0,1]. All boxes that have a Jaccard similarity of 
                  greater than `iou_threshold` with a locally maximal box will be 
                  removed from the set of predictions for a given class, 
                  where 'maximal' refers to the box's confidence score.
        * top_k (int, optional):
                - The number of highest scoring predictions to be kept for 
                  each batch item after the non-maximum suppression stage.
        * nms_max_output_size (int, optional):
                - The maximal number of predictions that will be left over after the NMS stage.
        * return_predictor_sizes (bool, optional):
                - If `True`, this function not only returns the model, but also
                  a list containing the spatial dimensions of the predictor layers. 
                  This isn't strictly necessary since you can always get their sizes easily 
                  via the Keras API, but it's convenient and less error-prone
                  to get them this way. They are only relevant for training anyway 
                  (SSDBoxEncoder needs to know the
                  spatial dimensions of the predictor layers), for inference you don't need them.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    #########################################################################
    # Thuc hien trien khai viec build model SSD300 bang Keras
    #########################################################################

    # So class duoc du doan trong SSD goc
    n_predictor_layers = 6
    # Cong them mot class bieu thi cho background
    n_classes += 1
    # Dat lai ten cho chuan hoa L2 duoc ngan gon
    l2_reg = l2_regularization
    # Lay kich thuoc cua hinh anh dau vao
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Dua ra mot vai ngoai le truoc khi thuc hien build models
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` va `aspect_ratios_per_layer` khong duoc de ca hai deu None. At least one needs to be specifiedIt nhat mot gia tri phai duoc chi dinh.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("No phai thuoc mot trong hai truong hop: aspect_ratios_per_layer la None hoac len(aspect_ratios_per_layer) == {}, nhung len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Ca `min_scale` va `max_scale` hoac `scales` can duoc chi dinh!")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("No phai thuoc mot trong hai truong hop scales la None hoac len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else:
        # Neu khong co list scale duoc truyen vao mot cach ro rang, thi se tinh cac scale thong qua hai gia tri min_scale va max_scale
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 gia tri variance phai duoc truyen vao nhung co {} gia tri duoc nhan!".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("Tat ca cac gia tri cua variance >0, nhung variances duoc nhan la {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("Ban phai cung cap it nhat mot gia tri step cho moi predictor layers!")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("Ban phai cung cap it nhat mot gia tri offset cho moi predictor layers!")

    ############################################################################
    # Tinh toan cac tham so cho Anchor Box
    ############################################################################

    # Dat ca aspect ratio (ty le khung hinh) cho moi lop predictor layer
    # Dieu nay la can thiet cho cac Anchor box layers
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Tinh toan so luong cac boxes duoc predicted tren moi cell cua moi predictor layers.
    # Chung ta can dieu nay de chung ta biet co bao nhieu channels ma cac predictor layers can phai co.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:
        # Neu chi co global aspect ratio list duoc truyen vao thi so luong boxes du doan
        # tren moi cell la nhu nhau doi voi cac predictor layers.
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Xac dinh cac function cho cac layer custom Lambda ben duoi
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build Network SSD (Xay dung mang SSD)
    ############################################################################

    # Layer dau vao (kich thuoc bang kich thuoc cua hinh anh duoc truyen vao)
    x = Input(shape=(img_height, img_width, img_channels))

    # Cac Layer tien xu ly hinh anh (chuan hoa hinh anh dau vao)

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    # Xay dung lai model VGG-16

    # Block 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    # Het model VGG-16
    
    #################################################################
    # Bat dau xay dung cac predictor layers
    #################################################################


    # Phan 1: Noi cac convolution voi nhau

    # FC6
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    # FC7
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    # Conv8_2
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    # Conv9_2
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    # Conv10_2
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    # Conv11_2
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # Phan 2: Tao cac predictor convolution

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    ### Build cac convolutional predictor layers tren top (dinh) cua base network


    #  Phan 2.1: Xay dung trinh layers confidences: do tin cay cho moi box du doan

    # Chung toi predict `n_clases` gia tri confidence cho moi box, do do cac confidence predictor co
    # do sau `n_boxes * n_classes`
    # Hinh dang dau ra cua layers confidences: `(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)

    # Phan 2.2: Xay dung phan predict toa do cho moi box du doan

    # Chung toi du doan 4 toa do cho moi box,
    # do do, cac yeu to predictor localization se co chieu sau la `n_boxes * 4`
    # Hinh dang dau ra cua layers localization: `(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

    # Phan 2.3: Tao cac Anchor boxs

    ### Tao ra cac anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Dau ra cua cac Anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # Dinh hinh lai shape cua class predictor, mang lai 3D tensors voi shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)

    # Reshape the box predictions, mang lai 3D tensors voi shape `(batch, height * width * n_boxes, 4)`
    # Chung toi muon 4 toa do se bi co lap (isolated) o truc cuoi (last axis) de co the tinh 
    # toan cho smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    # Reshape the anchor box tensors, mang lai 3D tensors voi shape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Noi cac predictions tu cac lop khac nhau lai voi nhau

    # Axis 0 (batch) va axis 2 (n_classes hoac 4, tuong ung) la giong het nhau cho tat ca cac layers,
    # vi vay chung toi muon noi doc theo truc axis 1, do la so luong box tren moi layers
    # Output shape cua `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape cua `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape cua `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # Cac predictor ve toa do se di vao ham loss function theo cach cua chung,
    # nhung con doi voi cac predict class, Chung ta se ap dung lop kich hoat softmax
    # truoc
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Nối các dự đoán của lớp (class) và box và các anchor vào một vectơ dự đoán lớn
    # Output shape cua `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                     fc7_mbox_conf._keras_shape[1:3],
                                     conv6_2_mbox_conf._keras_shape[1:3],
                                     conv7_2_mbox_conf._keras_shape[1:3],
                                     conv8_2_mbox_conf._keras_shape[1:3],
                                     conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
