'''
A data generator for 2D object detection.

'''

from __future__ import division
import numpy as np
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import cv2
import csv
import os
import sys
from tqdm import tqdm, trange
try:
    import h5py
except ImportError:
    warnings.warn("'h5py' module is missing. The fast HDF5 dataset option will be unavailable.")
try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass

class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass

class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides three methods to parse annotation data: A general-purpose CSV parser,
    an XML parser for the Pascal VOC datasets, and a JSON parser for the MS COCO datasets.
    If the annotations of your dataset are in a format that is not supported by these parsers,
    you could just add another parser method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 load_images_into_memory=False,
                 hdf5_dataset_path=None,
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        '''
        Initializes the data generator. You can either load a dataset directly here in the constructor,
        e.g. an HDF5 dataset, or you can use one of the parser methods to read in a dataset.

        Arguments:
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            hdf5_dataset_path (str, optional): The full file path of an HDF5 file that contains a dataset in the
                format that the `create_hdf5_dataset()` method produces. If you load such an HDF5 dataset, you
                don't need to use any of the parser methods anymore, the HDF5 dataset already contains all relevant
                data.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
            eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
                a list that indicates for each ground truth object in the image whether that object is supposed
                to be treated as neutral during an evaluation.
            labels_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
                strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
        '''
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.

        self.dataset_size = 0 # As long as we haven't loaded anything yet, the dataset size is zero.
        self.load_images_into_memory = load_images_into_memory
        self.images = None # The only way that this list will not stay `None` is if `load_images_into_memory == True`.

        # `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves. This list is one of the outputs of the parser methods.
        # In case you are loading an HDF5 dataset, this list will be `None`.
        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
            if load_images_into_memory:
                self.images = []
                if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else: it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))
        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
        # of ground truth bounding boxes for that image.
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

        if not hdf5_dataset_path is None:
            self.hdf5_dataset_path = hdf5_dataset_path
            self.load_hdf5_dataset(verbose=verbose)
        else:
            self.hdf5_dataset = None

    def load_hdf5_dataset(self, verbose=True):
        '''
        Tai mot tap du lieu HDF5 theo dinh dang ma phuong thuc create_hdf5_dataset()` tao ra

        Arguments:
            verbose (bool, optional): Neu true se in ra tien trinh khi tai bo du lieu

        Returns:
            None.
        '''

        # Doc du lieu tu file HDF5
        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        # Kich thuoc bo du lieu
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Thay vi xao tron bo du lieu HDF5 hoac hinh anh trong bo nho, chung toi se
        # xao tron du lieu bang chi muc nay
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

        # Neu co tuy chon load image vao bo nho
        if self.load_images_into_memory:
            self.images = []
            if verbose: tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else: tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose: tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else: tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose: tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else: tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose: tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else: tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        '''
        Cac doi so:
            images_dir (str): Duong dan den thu muc chua hinh anh.
            labels_filename (str): filepath cho mot tep CSV chua ground truth bounding box
                tren moi dong, va moi dong se chua 6 muc sau: ten file hinh anh, classID, xmin,
                xmax, ymin, ymax.
                Sau muc nay khong phai theo thu tu cu the, nhung chung phai la 6 cot dau tien
                cua moi dong.
                Thu tu cac tep nay trong CSV phai duoc chi dinh trong input_format.
                ID la mot so int > 0. Class ID 0 chi danh rieng cho class background.
                `xmin` va `xmax` la cac toa do ngang ben trai va ben phai nhat cua bbx.
                `ymin` va `ymax` la toa do doc tuyet doi nhat tren cung va duoi cung cua bbx.
                Ten hinh anh du kien se chi la ten cua tep hinh anh ma khong co duong dan thu muc
                ma hinh anh se duoc dat.
            input_format (list): Mot list 6 strings bieu dien thu tu cua 6 items cua hinh anh bao
                gom: ten tep hinh anh, classID, xmin, xmax, ymin, ymax, va classid.
            include_classes (list, optional): Hoac la all hoac la list cac so int chua cac ID
                cua cac class se duoc bao gom trong dataset. Neu la all, tat cac cac 
                ground truth boxes se duoc bao gom trong dataset.
            random_sample (float, optional): Hoac la False hoac la mot mang float [0, 1].
                Neu day la false, bo du lieu day du se duoc su dung boi trinh tao du lieu (data 
                generator). Neu day la so float [0, 1], mot phan duoc lay mau ngau nhien,
                cua tap du lieu se duoc su dung, trong do 'random_sample' la phan cua tap du 
                lieu duoc su dung. vi du, neu `random_sample = 0.2`, 20 phan tram cua bo du lieu se duoc chon, phan con lai se bi bo qua. The fraction de cap den so luong hinh anh,
                khong phai so luong box, tuc la moi hinh anh se duoc them vao bo du lieu se
                luon duoc them vao voi cac box cua no.
            ret (bool, optional): Co hay khong tra ve ket qua dau ra cua trinh phan tich cu phap.
            verbose (bool, optional): Neu true se in ra tien trinh cho cac hoat dong, va dieu
            nay co the lam lau qua trinh hon mot chut.

        Returns:
            Khong co mac dinh, list tuy chon cho bat ky hinh anh nao co san, ten tep hinh anh, labels, va ID cua hinh anh.
        '''

        # Thiet lap cac thanh phan cua mot class

        # duong dan den thu muc chua hinh anh
        self.images_dir = images_dir

        # duong dan den file csv
        self.labels_filename = labels_filename

        # mot list gom 6 yeu to: xmin, ymin, xmax, ymax, classID
        self.input_format = input_format # type: list

        # Co bao gom class khong
        self.include_classes = include_classes

        # Truoc khi bat dau, hay dam bao rang labels_filename va input_format khong phai
        # la None
        if self.labels_filename is None or self.input_format is None:
            raise ValueError("`labels_filename` va/hoac `input_format` chua duoc truyen vao. Ban can phai truyen vao chung voi cac gia tri phu hop.")

        # Xoa cac du lieu co the da duoc phan tich cu phap truoc do
        self.filenames = []
        self.image_ids = []
        self.labels = []

        # Dau tien, chi can doc cac dong cua tep CSV va sap xep chung

        # mang luu du lieu
        data = []

        # mo file csv va thuc hien doc file
        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            # bo qua hang tieu de
            next(csvread)
            # Doi voi dong (nghia la cho moi bounding box) trong tep CSV
            for row in csvread:
                # Neu co bao gom tat cac cac class va lay chi so class_id tuong ung trong
                # input_format.
                # Neu class_id nam trong so cac class duoc bao gom trong dataset:
                if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes:

                    # Luu tru cac box class va cac toa do tai day
                    box = []
                    # Chon cot chua ten cua hinh anh trong dinh dang dau vao input_format
                    # va noi dung cua no vao trong box
                    box.append(row[self.input_format.index('image_name')].strip())
                    # Doi voi moi thanh phan o dinh dang dau ra (trong do co cac phan tu la classID
                    # va 4 toa do cua bbx)
                    for element in self.labels_output_format:
                        # Chon cac cot tuong ung voi dinh dang dau vao va noi no cho box.
                        box.append(int(row[self.input_format.index(element)].strip()))
                    data.append(box)

        # Du lieu can duoc sap xep, neu khong thi buoc tiep theo se khong cho ket qua chinh xac
        data = sorted(data)
        # Bay gio chung ta da dam bao rang du lieu duoc sap xep theo ten tep, chung toi co the bien
        # dich cac danh sach mau vi du va nhan thuc te.

        # Hinh anh hien tai ma chung toi dang thu thap cac ground truth boxes
        current_file = data[0][0]

        # ID hinh anh se la mot phan cua ~ tuc la bo duoi jpg tu ten file hinh anh
        current_image_id = data[0][0].split('.')[0]

        # List noi chung toi thu thap tat ca cac ground truth boxes cho mot hinh anh nhat dinh
        current_labels = []
        add_to_dataset = False

        for i, box in enumerate(data):

            # i la index va box la data
            # Neu box nay (nghia la dong nay cua tep csv) thuoc ve hinh anh hien tai
            # do mot hinh anh co the co nhieu cac bbx cho nen ta phai xet cho tung hinh anh
            if box[0] == current_file:
                # append labels (bbx + class) vao trong current_labels
                current_labels.append(box[1:])
                # Neu day la dong cuoi cua tep csv
                if i == len(data)-1:
                    # trong tuong hop chung toi khong su dung bo du lieu day du, nhung chung
                    # toi se lay mot mau ngau nhien cua no
                    if random_sample:
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            # Neu box nay thuoc ve mot hinh anh moi
            else:

                # Trong truong hop chung toi khong su dung bo du lieu day du, 
                # nhung su dung cac mau ngau nhien cua no
                if random_sample:
                    p = np.random.uniform(0,1)
                    if p >= (1-random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                # Dat lai danh sach nhan vi day la tep moi
                current_labels = []
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        # trong truong hop chung toi muon tra lai
        if ret:
            return self.images, self.filenames, self.labels, self.image_ids

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes = 'all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False,
                  verbose=True):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels, image IDs,
            and a list indicating which boxes are annotated with the label "difficult".
        '''
        # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        # Xoa du lieu co the da duoc phan tich cu phap truoc
        self.filenames = []
        self.image_ids = []
        self.labels = []
        self.eval_neutral = []
        if not annotations_dirs:
            self.labels = None
            self.eval_neutral = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids

            if verbose: it = tqdm(image_ids, desc="Processing image set '{}'".format(os.path.basename(image_set_filename)), file=sys.stdout)
            else: it = image_ids

            # Loop over all images in this dataset.
            for image_id in it:

                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    # Parse the XML file for this image.
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                    #filename = soup.filename.text

                    boxes = [] # We'll store all boxes for this image here.
                    eval_neutr = [] # We'll store whether a box is annotated as "difficult" here.
                    objects = soup.find_all('object') # Get a list of all objects in this image.

                    # Parse the data for each object.
                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        # Check whether this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                        pose = obj.find('pose', recursive=False).text
                        truncated = int(obj.find('truncated', recursive=False).text)
                        if exclude_truncated and (truncated == 1): continue
                        difficult = int(obj.find('difficult', recursive=False).text)
                        if exclude_difficult and (difficult == 1): continue
                        # Get the bounding box coordinates.
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
                        item_dict = {'folder': folder,
                                     'image_name': filename,
                                     'image_id': image_id,
                                     'class_name': class_name,
                                     'class_id': class_id,
                                     'pose': pose,
                                     'truncated': truncated,
                                     'difficult': difficult,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult: eval_neutr.append(True)
                        else: eval_neutr.append(False)

                    self.labels.append(boxes)
                    self.eval_neutral.append(eval_neutr)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids, self.eval_neutral

    def parse_json(self,
                   images_dirs,
                   annotations_filenames,
                   ground_truth_available=False,
                   include_classes='all',
                   ret=False,
                   verbose=True):
        '''
        This is an JSON parser for the MS COCO datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the JSON format of the MS COCO datasets.

        Arguments:
            images_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for MS COCO Train 2014, another one for MS COCO
                Val 2014, another one for MS COCO Train 2017 etc.).
            annotations_filenames (list): A list of strings, where each string is the path of the JSON file
                that contains the annotations for the images in the respective image directories given, i.e. one
                JSON file per image directory that contains the annotations for all images in that directory.
                The content of the JSON files must be in MS COCO object detection format. Note that these annotations
                files do not necessarily need to contain ground truth information. MS COCO also provides annotations
                files without ground truth information for the test datasets, called `image_info_[...].json`.
            ground_truth_available (bool, optional): Set `True` if the annotations files contain ground truth information.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. If 'all', all ground truth boxes will be included in the dataset.
            ret (bool, optional): Whether or not to return the outputs of the parser.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally lists for whichever are available of images, image filenames, labels and image IDs.
        '''
        self.images_dirs = images_dirs
        self.annotations_filenames = annotations_filenames
        self.include_classes = include_classes
        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        if not ground_truth_available:
            self.labels = None

        # Build the dictionaries that map between class names and class IDs.
        with open(annotations_filenames[0], 'r') as f:
            annotations = json.load(f)
        # Unfortunately the 80 MS COCO class IDs are not all consecutive. They go
        # from 1 to 90 and some numbers are skipped. Since the IDs that we feed
        # into a neural network must be consecutive, we'll save both the original
        # (non-consecutive) IDs as well as transformed maps.
        # We'll save both the map between the original
        self.cats_to_names = {} # The map between class names (values) and their original IDs (keys)
        self.classes_to_names = [] # A list of the class names with their indices representing the transformed IDs
        self.classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
        self.cats_to_classes = {} # A dictionary that maps between the original (keys) and the transformed IDs (values)
        self.classes_to_cats = {} # A dictionary that maps between the transformed (keys) and the original IDs (values)
        for i, cat in enumerate(annotations['categories']):
            self.cats_to_names[cat['id']] = cat['name']
            self.classes_to_names.append(cat['name'])
            self.cats_to_classes[cat['id']] = i + 1
            self.classes_to_cats[i + 1] = cat['id']

        # Iterate over all datasets.
        for images_dir, annotations_filename in zip(self.images_dirs, self.annotations_filenames):
            # Load the JSON file.
            with open(annotations_filename, 'r') as f:
                annotations = json.load(f)

            if ground_truth_available:
                # Create the annotations map, a dictionary whose keys are the image IDs
                # and whose values are the annotations for the respective image ID.
                image_ids_to_annotations = defaultdict(list)
                for annotation in annotations['annotations']:
                    image_ids_to_annotations[annotation['image_id']].append(annotation)

            if verbose: it = tqdm(annotations['images'], desc="Processing '{}'".format(os.path.basename(annotations_filename)), file=sys.stdout)
            else: it = annotations['images']

            # Loop over all images in this dataset.
            for img in it:

                self.filenames.append(os.path.join(images_dir, img['file_name']))
                self.image_ids.append(img['id'])

                if ground_truth_available:
                    # Get all annotations for this image.
                    annotations = image_ids_to_annotations[img['id']]
                    boxes = []
                    for annotation in annotations:
                        cat_id = annotation['category_id']
                        # Check if this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not cat_id in self.include_classes): continue
                        # Transform the original class ID to fit in the sequence of consecutive IDs.
                        class_id = self.cats_to_classes[cat_id]
                        xmin = annotation['bbox'][0]
                        ymin = annotation['bbox'][1]
                        width = annotation['bbox'][2]
                        height = annotation['bbox'][3]
                        # Compute `xmax` and `ymax`.
                        xmax = xmin + width
                        ymax = ymin + height
                        item_dict = {'image_name': img['file_name'],
                                     'image_id': img['id'],
                                     'class_id': class_id,
                                     'xmin': xmin,
                                     'ymin': ymin,
                                     'xmax': xmax,
                                     'ymax': ymax}
                        box = []
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                    self.labels.append(boxes)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose: it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else: it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:
            return self.images, self.filenames, self.labels, self.image_ids

    def create_hdf5_dataset(self,
                            file_path='dataset.h5',
                            resize=False,
                            variable_image_size=True,
                            verbose=True):
        '''
        Chuyen doi tap du lieu hien tai dang tai thanh tap du lieu trong dinh dang HDF5.
        Tep HDF5 nay chua tat ca cac hinh anh duoi dang mang khong nen trong mot khoi bo 
        nho lien ke, cho phep chung tai nhanh hon. 
        Tuy nhien, mot bo du lieu khong nen nhu vay co the chiem nhieu dung luong dang ke
        tren o cung cua ban so voi tong so hinh anh nguon o dinh dang nen JPG va PNG.

        Ban nen luon luon chuyen doi tap du lieu thanh tap du lieu duoc luu trong dinh dang
        HDF5 neu ban co du dung luong o cung. Viec tai du lieu tu trong tep HDF5 se giup
        tang toc do dang ke trong viec load du lieu.

        Luu y la ban phai tai du lieu thong qua mot trinh phan tich cu phap truoc khi tao bo du
        HDF5 tu du lieu tra ve cua trinh phan tich cu phap.

        Bo du lieu HDF5 da tao se van duoc mo de co the su dung duoc ngay sau khi tao no.

        Cac doi so:
            file_path (str, optional): Duong dan day du de luu tru file HDF5. Ban co the load
                tep dau ra nay thong qua contructor DataGenerator trong tuong lai
            resize (tuple, optional): Flase hoac 2 tuple (chieu cao, chieu rong) dai dien cho
                kich thuoc dich cua hinh anh. Tat ca hinh anh trong bo du lieu se duoc thay doi kich thuoc theo kich thuoc muc tieu nay truoc khi chung duoc ghi vao tep HDF5.
                Neu False, khong thay doi kich thuoc se duoc thuc hien.
            variable_image_size (bool, optional): Muc dich duy nhat cua doi so nay la gia tri cua
                no se duoc luu tru trong bo du lieu HDF5 de co the nhanh chong tim hieu xem tat ca
                cac hinh anh trong bo du lieu co cung kich thuoc hay khong.
            verbose (bool, optional): Co hay khong in ra tien trinh tao du lieu

        Returns:
            None.
        '''

        # duong dan day du luu tru file HDF5
        self.hdf5_dataset_path = file_path

        # kich thuoc cua dataset ~ so luong cac sample trong dataset
        dataset_size = len(self.filenames)

        # Tao ra file HDF5
        hdf5_dataset = h5py.File(file_path, 'w')

        # Tao mot vai thuoc tinh cho chung ta biet bo du lieu nay chua nhung gi.
        # Bo du lieu ro rang se luon chua hinh anh, nhung no cung co the chua labels, ID cua hinh
        # anh, .. 
        hdf5_dataset.attrs.create(name='has_labels', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_image_ids', data=False, shape=None, dtype=np.bool_)
        hdf5_dataset.attrs.create(name='has_eval_neutral', data=False, shape=None, dtype=np.bool_)
        # That huu ich khi co the nhanh chong kiem tra xem tat ca cac hinh anh trong bo du lieu
        # trong bo du lieu co cung kich thuoc hay khong, vi vay hay them thuoc tinh boolean cho
        # dieu do.
        if variable_image_size and not resize:
            hdf5_dataset.attrs.create(name='variable_image_size', data=True, shape=None, dtype=np.bool_)
        else:
            hdf5_dataset.attrs.create(name='variable_image_size', data=False, shape=None, dtype=np.bool_)

        # Tao du lieu trong do cac hinh anh se duoc luu tru duoi dang cac mang phang.
        # Dieu nay cho phep chung toi, trong so nhung thu khac, luu tru hinh anh co kich thuoc thay doi
        hdf5_images = hdf5_dataset.create_dataset(name='images',
                                                  shape=(dataset_size,),
                                                  maxshape=(None),
                                                  dtype=h5py.special_dtype(vlen=np.uint8))

        # Tao tap du lieu se giu chieu cao, vhieu rong va kenh ma chung ta se can de tai tao
        # lai cac hinh anh tu mang duoc lam phang sau nay
        hdf5_image_shapes = hdf5_dataset.create_dataset(name='image_shapes',
                                                        shape=(dataset_size, 3),
                                                        maxshape=(None, 3),
                                                        dtype=np.int32)

        # Neu labels khong phai la None
        if not (self.labels is None):

            # Tao dataset trong do labels duoc luu tru duoi dang cac mang
            hdf5_labels = hdf5_dataset.create_dataset(name='labels',
                                                      shape=(dataset_size,),
                                                      maxshape=(None),
                                                      dtype=h5py.special_dtype(vlen=np.int32))

            # Tap tap du lieu se giu kich thuoc cua cac mang labels cho moi hinh anh de chung ta
            # co the khoi phuc cac labels tu cac mang da duoc lam phang sau do
            hdf5_label_shapes = hdf5_dataset.create_dataset(name='label_shapes',
                                                            shape=(dataset_size, 2),
                                                            maxshape=(None, 2),
                                                            dtype=np.int32)

            # update gia tri has_labels thanh True
            hdf5_dataset.attrs.modify(name='has_labels', value=True)

        # Neu co cac Class ID
        if not (self.image_ids is None):

            # Tao du lieu luu tru kich thuoc cua cac ClassID tuong ung voi dataset_size
            hdf5_image_ids = hdf5_dataset.create_dataset(name='image_ids',
                                                         shape=(dataset_size,),
                                                         maxshape=(None),
                                                         dtype=h5py.special_dtype(vlen=str))

            # Update gia tri cua has_image_ids thanh True
            hdf5_dataset.attrs.modify(name='has_image_ids', value=True)
        # 
        if not (self.eval_neutral is None):

            # Tao dataset trong do co cac labels se duoc luu tru duoi dang phang
            hdf5_eval_neutral = hdf5_dataset.create_dataset(name='eval_neutral',
                                                            shape=(dataset_size,),
                                                            maxshape=(None),
                                                            dtype=h5py.special_dtype(vlen=np.bool_))

            hdf5_dataset.attrs.modify(name='has_eval_neutral', value=True)

        # Co hay khong in ra cac tien trinh xu ly
        if verbose:
            tr = trange(dataset_size, desc='Creating HDF5 dataset', file=sys.stdout)
        else:
            tr = range(dataset_size)

        # Lap qua tat ca cac hinh anh co trong bo du lieu
        for i in tr:

            # Luu tru hinh anh
            with Image.open(self.filenames[i]) as image:

                # chuyen hinh anh ve dang array
                image = np.asarray(image, dtype=np.uint8)

                # Hay chac chan rang hinh anh cuoi cung se co ba channels
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    if image.shape[2] == 1:
                        image = np.concatenate([image] * 3, axis=-1)
                    elif image.shape[2] == 4:
                        image = image[:,:,:3]
                # Neu phai dinh hinh lai hinh anh ve kich thuoc co dinh
                if resize:
                    image = cv2.resize(image, dsize=(resize[1], resize[0]))

                # Lam phang mang hinh anh va ghi no vao bo du lieu hinh anh
                hdf5_images[i] = image.reshape(-1)
                # Viet hinh dang cua hinh anh vao bo du lieu hinh dang cua hinh anh
                hdf5_image_shapes[i] = image.shape

            # Luu tru nhan cua cac bbx neu co bat ky cac labels nao do hop le
            if not (self.labels is None):

                # Chuyen labels ve dang array
                labels = np.asarray(self.labels[i])
                # Lam phang mang labels va ghi no vao tap du lieu labels
                hdf5_labels[i] = labels.reshape(-1)
                # Viet hinh dang cua labels vao tap du lieu hinh dang cua labels
                hdf5_label_shapes[i] = labels.shape

            # Luu tru ID cua hinh anh neu chung ta co bat ky classID cua hinh anh nao do
            if not (self.image_ids is None):

                hdf5_image_ids[i] = self.image_ids[i]

            # Luu tru cac chu thich trung lap neu chung ta co bat ky thong tin gi ve no
            if not (self.eval_neutral is None):

                hdf5_eval_neutral[i] = self.eval_neutral[i]

        hdf5_dataset.close()
        self.hdf5_dataset = h5py.File(file_path, 'r')
        self.hdf5_dataset_path = file_path
        self.dataset_size = len(self.hdf5_dataset['images'])
        # Thay vi xao tron bo du lieu HDF5, chung toi se xao tron danh sach chi muc nay,
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        Tao ra cac batches cua cac samples va (tuy chon) nhan tuong ung.

        Co the xao tron cac samples nhat quan sau moi lan vuot qua mot cach hoan chinh.

        Tuy chon lay mot danh sach cac bien doi hinh anh tuy y de ap dung cho cac mau ad hoc.

        Cac doi so:
            * batch_size (int, optional):
                - Kich thuoc cac batch se duoc tao ra.
            * shuffle (bool, optional):
                - Co hay ko xao tron du lieu moi lan pass qua. Tuy chon nay luon luon phai la
                  True trong qua trinh dao tao. nhung cung co the huu ich de tat xao tron
                  de go loi hoac ban dang su dung trinh du doan.
            * transformations (list, optional):
                - Mot danh sach cac bien doi se duoc ap dung cho cac hinh anh va nhan theo
                  thu tu nhat dinh. Moi phep bien doi la mot lenh co the goi dau vao la mot
                  hinh anh dang numpy va nhan cung la dang numpy va tra ve mot hinh anh nhan
                  tuy chon co cung dinh dang.
            * label_encoder (callable, optional):
                - Chi lien quan den labels duoc dua ra. Mot cuoc goi lay dau vao la nhan cua mot
                  batch (duoi dang mang numpy) va tra ve mot cau truc dai dien cho cac labels do.
                  Truong hop su dung chung cho viec nay la chuyen tu dinh dang labels dau vao
                  sang dinh dang ma model detect can de lam muc tieu dao tao cho no.
            * returns (set, optional):
                Mot tap hop cac chuoi xac dinh nhung gi dau ra ma trinh tao mang lai.
                Dau ra cua trinh tao luon la mot tuple chua cac dau ra duoc chi dinh trong bo nay
                va chi cac dau ra. Neu mot dau ra khong co san, no se la None, Bo du lieu dau 
                ra co the chua cac dau ra sau theo cac chuoi tu khoa da chi dinh:
                * 'processed_images':
                    - Mot mang chua cac hinh anh da duoc xu ly. Se luon co cac dau ra, vi vay viec
                      ban co bao gom tu khoa nay trong tap hop hay khong khong quan trong.
                * 'encoded_labels':
                    - Cac tensor duoc encode. se luon o trong dau ra neu label encoder duoc cung
                      cap, do do viec ban co bao gom tu khoa nay trong tap hop hay khong neu ban
                      vuot qua bo du lieu
                * 'matched_anchors':
                    - Chi kha dung neu labels_encoder la doi tuong trong SSDInputEncoder
                    - Giong nhu encoded_labels, nhung chua toa do cac anchor box cho tat ca cac
                      anchor box phu hop thay vi toa do cac ground truth.
                    - Dieu nay co the huu ich de hinh dung nhung anchor box nay dang khop voi
                      ground truth. Chi co san trong che do trainning.
                * 'processed_labels':
                    - Cac nhan duoc xu ly, nhugn chua duoc ma hoa. Day la danh sach chua cho moi
                      hinh anh hang loat mang numpy voi tat ca cac ground truth cho hinh anh do.
                    - Chi co san neu ground truth co san.
                * 'filenames':
                    - Mot danh sach chua cac ten tep (duong dan day du) cho hinh anh trong batches.
                * 'image_ids':
                    - Mot danh sach chua ID cac so nguyen cua cac hinh anh trong batches. Chi kha
                      dung neu co ID hinh anh co san.
                * 'evaluation-neutral':
                    - Mot danh sach long nhau cua danh sach cac booleans. Moi danh sach chua cac
                      gia tri True hoac False cho moi  ground truth bounding box cua hinh anh tuong ung tuy thuoc vao viec bbx co duoc danh gia la trung lap (true) hay None (False).
                    - Co the tra ve None neu ko ton tai khai niem nhu vay cho mot tap du lieu da
                      cho. Mot vi du cho tinh trung lap do la cac  ground truth bounding box duoc
                      chu thich la kho trong bo du lieu Pascal VOC, thuong duoc coi la trung tinh
                      trong danh gia model.
                * 'inverse_transform':A nested list that contains a list of "inverter" functions for each item in the batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                    that were applied to the original image to them. This makes it possible to let the model make predictions on a
                    transformed image and then convert these predictions back to the original image. This is mostly relevant for
                    evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                    predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                    original images, not with respect to the transformed images. This means you will have to transform the predicted
                    box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                    image need to be applied in the order in which they are given in the respective list for that image.
                * 'original_images':
                    - Mot danh sach cac hinh anh goc trong batch truoc khi duoc xu ly.
                * 'original_labels':
                    - Mot danh sach chua cac ground truth boxes ban dau cho cac hinh anh trong
                      batch nay truoc khi duoc xu ly. Chi co san neu ground truth co san.
                    - Thu tu dau cua cac dau ra trong tuple la thu tu cua danh sach tren.
                    - Neu returns chua mot tu khoa cho mot dau ra ko co san, dau ra do se bi bo qua
                      trong cac bo du lieu mang lai va canh bao se duoc dua ra.
            * keep_images_without_gt (bool, optional):
                - Neu false cac hinh anh ko co ground truth boxes truoc khi ap dung bat ky bien
                  doi nao se bi xoa khoi batch. Neu True, nhung hinh anh nhu vay se duoc giu lai
                  trong batches.
            * degenerate_box_handling (str, optional): 
                - cach xu ly cac hop handle, cac hop co xmax <= xmin va/hoac ymax <= ymin 
                - Các hộp thoái hóa kieu nhu the nay đôi khi có thể nằm trong bộ dữ liệu 
                  hoặc các hộp không suy biến có thể bị thoái hóa sau khi chúng được xử lý 
                  bằng các phép biến đổi.
                - Lưu ý rằng trình tạo kiểm tra các hộp suy biến sau khi tất cả các 
                  phép biến đổi đã được áp dụng (nếu có), nhưng trước khi các nhãn được 
                  chuyển đến `label_encoder` (nếu được đưa ra).
                - Có thể là một trong những 'cảnh báo' hoặc 'loại bỏ'. 
                  Nếu 'cảnh báo', trình tạo sẽ chỉ in cảnh báo để cho bạn biết rằng có các 
                  hộp thoái hóa trong một lô. Nếu 'loại bỏ', trình tạo sẽ loại bỏ các hộp 
                  thoái hóa khỏi lô một cách im lặng.

        Yields:
            Cac batch tiep theo duoi dang mot tuple cac items nhu duoc dinh nghia boi doi so
            returns.
        '''

        # # Neu kich thuoc cua dataset bang khong
        # if self.dataset_size == 0:
        #     raise DatasetError("Khong the tao ra cac batch vi ban khong load data!")

        # #############################################################################################
        # # Canh bao neu bat ky tap hop nao tra ve la khong the
        # #############################################################################################
        # # neu labels la None
        # if self.labels is None:
        #     if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
        #         warnings.warn("Ke tu khi khong co labels nao duoc dua ra, khong co gia tri nao trong so 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', va 'matched_anchors'" +
        #                       "la cac gia tri hop le, nhung ban dat `returns = {}`. Khong the nao tra ve la None".format(returns))
        # elif label_encoder is None:
        #     if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
        #         warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
        #                       "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        # elif not isinstance(label_encoder, SSDInputEncoder):
        #     if 'matched_anchors' in returns:
        #         warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
        #                       "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        # Lam mot vai dieu de chuan bi nhu co the xao tron bo du lieu ban dau
        #############################################################################################

        # Neu duoc phep xao tron du lieu
        if shuffle:
            # Lay cac chi so (index da duoc tao ra san cho viec xao tron du lieu)
            objects_to_shuffle = [self.dataset_indices]

            # Neu co filename
            if not (self.filenames is None):
                # them filename vao trong mang xao tron du lieu
                objects_to_shuffle.append(self.filenames)
            # Neu co labels
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            # Neu co image_ids
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            # Xao tron hinh anh
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)

            # Thuc hien chay vong lap chay qua mang objects_to_shuffle
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Ghi de cac dinh dang labels cua tat ca cac phep bien doi de dam bao chung co 
        # the duoc dat mot cach chinh xac
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # Tao ra cac mini_batches
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = 0

            #########################################################################################
            # Có thể xáo trộn tập dữ liệu nếu vượt qua toàn bộ tập dữ liệu.
            #########################################################################################

                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            #########################################################################################
            # Lấy hình ảnh, (có thể) ID hình ảnh, (có thể) nhãn, v.v. cho lô này.
            #########################################################################################

            # Chúng tôi ưu tiên các tùy chọn của mình theo thứ tự sau:
            # 1) Nếu chúng ta có những hình ảnh đã được tải trong bộ nhớ, hãy lấy chúng từ đó.
            # 2) Khác, nếu chúng ta có một bộ dữ liệu HDF5, hãy lấy hình ảnh từ đó.
            # 3) Khác, nếu chúng ta không có những điều trên, chúng ta sẽ phải tải các tệp hình ảnh riêng lẻ từ đĩa.
            batch_indices = self.dataset_indices[current:current+batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            elif not (self.hdf5_dataset is None):
                for i in batch_indices:
                    batch_X.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current+batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            # Lấy nhãn cho lô này (nếu có).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current+batch_size]
            else:
                batch_eval_neutral = None

            # Lấy ID hình ảnh cho đợt này (nếu có).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # Các hình ảnh gốc, không thay đổi
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # Các nhãn gốc, không thay đổi

            current += batch_size

            #########################################################################################
            # Có thể thực hiện chuyển đổi hình ảnh.
            #########################################################################################

            batch_items_to_remove = [] # Trong trường hợp chúng tôi cần xóa bất kỳ hình ảnh nào khỏi lô, lưu trữ các chỉ số của chúng trong danh sách này.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Chuyen doi labels cho hinh anh nay thanh mot mang (trong truong hop chung chua co)
                    batch_y[i] = np.array(batch_y[i])
                    # Neu hinh anh nay khong co ground truth boxes, co le chung ta se khong muon
                    # giu chung trong batches
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Ap dung bat ky bien doi hinh anh ma chung toi co da nhan duoc
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Kiem tra cac vi tri toa do ma bi danh nhan sai: xmax <= xmin, ymax <= ymin
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Xoa bat ky box nao chung toi co the ko muon giu lai trong batches
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # Dieu nay khong hieu qua, hy vong ban khong nen thuc hien thuong xuyen
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # THAN TRONG: Chuyen doi Batch_X thanh mot mang se dan den mot batch trong neu hinh anh
            # co kich thuoc khac nhau hoac so luong channels khac nhau. Tai thoi diem nay, tat ca
            # cac hinh anh phai co cung kich thuoc va so luong channels.
            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")

            #########################################################################################
            # Neu chung ta co tuy chon label_encoder thi hay thuc hien encode labels do
            #########################################################################################

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Soan cac ket qua dau ra
            #########################################################################################

            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            yield ret

    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None,
                     eval_neutral_path=None):
        '''
        Writes the current `filenames`, `labels`, and `image_ids` lists to the specified files.
        This is particularly useful for large datasets with annotations that are
        parsed from XML files, which can take quite long. If you'll be using the
        same dataset repeatedly, you don't want to have to parse the XML label
        files every time.

        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
            image_ids_path (str, optional): The path under which to save the image IDs pickle.
            eval_neutral_path (str, optional): The path under which to save the pickle for
                the evaluation-neutrality annotations.
        '''
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if not labels_path is None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if not image_ids_path is None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)
        if not eval_neutral_path is None:
            with open(eval_neutral_path, 'wb') as f:
                pickle.dump(self.eval_neutral, f)

    def get_dataset(self):
        '''
        Returns:
            4-tuple containing lists and/or `None` for the filenames, labels, image IDs,
            and evaluation-neutrality annotations.
        '''
        return self.filenames, self.labels, self.image_ids, self.eval_neutral

    def get_dataset_size(self):
        '''
        Returns:
            So luong cac hinh anh trong bo du lieu
        '''
        return self.dataset_size
