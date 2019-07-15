'''
Mot encoder chuyen doi ground truth annotations (cac chu thich cua bbx nhan) thanh
muc tieu dao tao tuong thich voi SSD

'''

from __future__ import division
import numpy as np

from bounding_box_utils.bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class SSDInputEncoder:
    '''
    Chuyen doi cac ground truth labels cho detect object trong hinh anh (toa do cac bbx
    2D va cac class labels) thanh dinh dang can thiet de dao tao model SSD.

    Trong qua trinh encoding cac ground truth labels, mot template cac anchor box dang
    duoc xay dung, sau do no duoc khop voi cac ground truth boxes thong qua chi so
    IoU giao nhau
    '''

    def __init__(self, img_height, img_width, n_classes, predictor_sizes, min_scale=0.1, max_scale=0.9, scales=None, aspect_ratios_global=[0.5, 1.0, 2.0], aspect_ratios_per_layer=None, two_boxes_for_ar1=True, steps=None, offsets=None, clip_boxes=False, variances=[0.1, 0.1, 0.2, 0.2], matching_type='multi', pos_iou_threshold=0.5, neg_iou_limit=0.3, border_pixels='half', coords='centroids', normalize_coords=True, background_id=0):
        '''
        Cac doi so:
            * img_height (int): Chieu cao cua hinh anh dau vao
            * img_width (int): Chieu rong cua hinh anh dau vao
            * n_classes (int):
                - So luong cac class positives (cac class object, ko phai class background)
                  vi du: 20 cho Pascal VOC, 80 cho MS COCO.
            * predictor_sizes (list):
                - Mot list cac so int theo dinh dang `(height, width) chua chieu cao va chieu
                  rong cua dau ra cua cac lop convolutional predictor layers.
            * min_scale (float, optional):
                - He so ty le nho nhat cho cac anchor box va no thuoc layer predict thap nhat.
                - He so nay phai > 0
                - He so nay phai duoc chon sao cho viec tao ra cac anchorbox tuong ung co cung
                  kich thuoc voi doi tuong can duoc phat hien.
            * max_scale (float, optional):
                - He so ty le lon nhat cho cac anchor box va no thuoc layer predict cao nhat. 
                - Tat ca cac he so ty le cho cac anchor box cua cac layer predict o giua hai
                  lop predict cao nhat va thap nhat se duoc noi suy tuyen tinh trong doan 
                  tu [min_scale, max_scale]
                - Ghi nho rang, he so ty le duoc noi suy tuyen tinh tu thu 2 tu cuoi cung se thuc su su la he so ty le cho layer predict cuoi cung, trong khi he so 
                ty le cuoi cung se duoc su dung cho second box cua aspect ratios = 1 
                trong layer predict cuoi cung, neu `two_boxes_for_ar1` duoc dat la `True`.
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
                - None hoac la mot danh sach co so luong phan tu bang voi so luong cua
                  predictor layers.
                - Cac phan tu co the la int/float
                - Nhung con so nay dai dien cho viec moi lop du doan co bao nhieu pixels
                  ngoai cac diem trung tam cua cac anchor box phai theo chieu doc va chieu
                  ngang doc theo luoi khong gian cua hinh anh.
                - Neu danh sach chua ints/floats, thi gia tri do se duoc su dung cho ca hai kich
                  thuoc khong gian.
                - Neu danh sach chua cac tuples cua hai ints/floats thi chung se dai dien cho
                  (step_height, step_width)
                - Neu ko co steps nao duoc cung cap, thi chung se duoc tinh toan sao cho cac
                  diem trung tam cua anchor box se tao thanh mot luoi cac deu kich thuoc hinh
                  anh.
            * offsets (list, optional):
                - None hoac la mot danh sach co so luong phan tu bang voi so luong cua
                  predictor layers.
                - Cac phan tu co the la float hoac la tuple cua floats.
                - Nhung con so nay dai dien cho moi predictor layers co bao nhieu pixel tuyet
                  doi, ma la mot phan cua kich thuoc step duoc chi dinh trong doi so steps.
                - Neu list chua cac so float, thi cac gia tri do duoc su dung cho ca hai
                  chieu khong gian. Neu danh sach co chua bo tuple cua floats hoac la ints
                  thi chung dai dien cho (vertical_offset, horizontal_offset)
                  Neu ko co offset duoc truyen vao thi chung se duoc mac dinh la 0.5 cho kich thuoc cua step (step size).
            * clip_boxes (bool, optional):
                - Neu la "true", gioi han toa do cua cac anchor box se nam trong ranh gioi
                  cua hinh anh
            * variances (list, optional):
                - Mot list 4 so float > 0. Anchor box offset cho moi toa do se duoc chia cho
                  gia tri variances tuong ung cua no.
            * matching_type (str, optional):
                - Co the la multi hoac la bipartite.
                - Trong che do bipartite, mot ground truth box se duoc match voi 1 anchor
                  box co do chong cheo IoU cao nhat
                - Trong che do multi, ngoai viec ket hop bipartite noi tren, tat ca cac 
                  anchor box co IoU trung hoac lon hon pos_iou_threshold se duoc matching
                  voi ground truth box de su dung trong loss function.
            * pos_iou_threshold (float, optional):
                - nguong IoU duoc su dung de khop hop anchor box va ground truth box nhat dinh.
            * neg_iou_limit (float, optional):
                - do tuong giao toi da cho phep cua anchor box voi bat ky ground truth box duoc
                  dan nhan la negative (vi du background). Neu anchor box khong phai la hop
                  duong, cung ko phai la box am thi se duoc bo qua trong qua trinh dao tao.
            * border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            * coords (str, optional):
                - Dinh dang toa do duoc su dung ben trong boi model (nghia la day ko phai dinh
                dang dau vao cua ground truth labels).
                - Co the la centroid voi dinh dang: cx, cy, w, h (toa do trung tam cua box,
                chieu rong va chieu cao cua box)
                - Co the la minmax (xmin, xmax, ymin, ymax) hoac cornner (goc) theo dinh
                dang (xmin, ymin, xmax, ymax)
            * normalize_coords (bool, optional):
                - Neu la true, encoder se su dung toa do tuong doi thay vi toa do tuyet doi
                Dieu nay co nghia la thay vi su dung toa do tuyet doi, encoder se chia lai
                ty le cac toa do khong gian ve doan [0, 1].
                - Cach hoc nay se tro nen doc lap voi kich thuoc hinh anh dau vao.
            * background_id (int, optional):
                - Xac dinh ID cho cac lop background
        '''

        # Kich thuoc cua cac lop predictor
        predictor_sizes = np.array(predictor_sizes)

        # Dinh hinh lai kich thuoc cua predictor size neu no khong o dinh dang mong muon
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # Dua ra mot so ngoai le.
        ##################################################################################

        # Phai truyen vao mot trong hai truong hop
        # + hoac la truyen ca min_scale
        # + hoac la truyen ca max_scale
        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Can phai chi dinh min_scale va max_scale hoac scales")

        # Neu truyen vao scales
        if scales:
            # Kiem tra neu scales ko bang so luong cac lop predictor + 1 (1 la scale
            # cho ratio = 1)
            if (len(scales) != predictor_sizes.shape[0] + 1):
                raise ValueError("No phai thuoc mot trong hai truong hop, hoac la None hoac la len(scales) == len(predictor_sizes)+1, nhung len(scales) == {} va len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes)+1))
            scales = np.array(scales)
            # Neu co bat ky gia tri scale nao <= 0
            if np.any(scales <= 0):
                raise ValueError("Tat ca cac gia tri trong `scales` phai lon hon 0, nhung list scale duoc truyen vao la: {}".format(scales))
        else:
            # Neu ko co list scale nao duoc truyen vao thi chung ta can phai dam bao rang
            # min_scale va max_scale phai la cac gia tri hop le
            if not 0 < min_scale <= max_scale:
                raise ValueError("Gia tri nay phai thoa man 0 < min_scale <= max_scale, nhung no lai la min_scale = {} va max_scale = {}".format(min_scale, max_scale))

        # Neu aspect ratio cho tung lop duoc truyen vao
        if not (aspect_ratios_per_layer is None):
            # Neu so luong cac ratio cho tung lop khong bang so luong cac lop predictor layers
            if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]):
                raise ValueError("No phai thuoc mot trong hai truong hop aspect_ratios_per_layer la None hoac len(aspect_ratios_per_layer) == len(predictor_sizes), nhung len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
            # Lap qua tung ratios dat cho tung lop predictor layers
            for aspect_ratios in aspect_ratios_per_layer:
                # Neu co bat ky ratio nao <= 0 thi no se dua ra ngoai le
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("Tat ca cac gia tri aspect ratios phai lon hon 0!")
        else:
            # Neu gia tri aspect ratios global khong duoc truyen vao
            if (aspect_ratios_global is None):
                raise ValueError("It nhat mot trong `aspect_ratios_global` va `aspect_ratios_per_layer` khong duoc mang gia tri `None`.")
            # Neu co bat ky gia tri aspect ratios global nao ma <= 0
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("Tat ca cac gia tri aspect ratios phai lon hon 0!")

        # Neu so luong variances khac 4
        if len(variances) != 4:
            raise ValueError("4 gia tri variance phai duoc truyen vao, nhung {} gia tri da duoc nhan".format(len(variances)))
        # Chuyen variances ve dang array
        variances = np.array(variances)

        # Neu co bat ky gia tri variances nao nho hon 0
        if np.any(variances <= 0):
            raise ValueError("Tat ca cac gia tri variances >0, nhung cac gia tri variances duoc dua ra la: {}".format(variances))

        # Neu cac gia tri toa do khong theo 1 trong ba dinh dang sau thi dua ra ngoai le
        if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
            raise ValueError("Day la gia tri khong hop le doi voi `coords`. Cac gia tri duoc ho tro do la 'minmax', 'corners' va 'centroids'.")

        # Neu steps ko None va kich thuoc cua step ko bang so luong cac lop predictor layers
        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError("Ban can phai cung cap it nhat mot gia tri steps cho moi lop predictor")

        # Neu offset duoc truyen vao va so luong offset ko bang so luong cac lop predictor layers
        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("Ban can cung cap it nhat mot gia tri offsets cho moi lop predictor")

        ##################################################################################
        # Thiet lap va tinh toan mot so yeu to
        ##################################################################################

        # Chieu cao cua hinh anh
        self.img_height = img_height

        # Chieu rong cua hinh anh
        self.img_width = img_width

        # So luong cac class bao gom ca background class
        self.n_classes = n_classes + 1

        # So luong cac predictor layers
        self.predictor_sizes = predictor_sizes

        # Cac he so ty le min, max scale
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Neu scale la None, hay tinh cac scale bang cach tinh noi suy tuyen tinh giua
        # min_scale va max_scale. Tuy nhien neu mot list ro rang cac scale duoc truyen vao
        # thi cac gia tri min_scale va max_scale se duoc ghi de.

        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)
        else:
            # Neu mot list cac scale duoc truyen vao mot cach ro rang, chung ta 
            # se thay no vao thay vi tinh toan min_scale hoac max_scale.
            self.scales = scales
        
        # Neu aspect_ratios_per_layer = None, thi chung ta se su dung cac ratio
        # global cho tat ca cac lop du doan, tuy nhien viec su dung cac ratios cho tung
        # layer se huu dung hon rat nhieu
        if (aspect_ratios_per_layer is None):
            # Clone ra cac aspect_ratios_global tuong ung bang so luong cac predictor layers
            # vi du: [[None]] * 4 = [[None], [None], [None], [None]]
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            # Neu cac aspect ratios duoc dua ra cho tung layer, chung ta se su dung ratios do
            self.aspect_ratios = aspect_ratios_per_layer

        # Hai box duoc tao ra khi aspect ratios = 1
        self.two_boxes_for_ar1 = two_boxes_for_ar1

        # Neu steps ma duoc truyen vao
        if not (steps is None):
            self.steps = steps
        else:
            # Tao ra mot mang chua so luong None bang so luong lop predictor layer
            self.steps = [None] * predictor_sizes.shape[0]

        # neu offsets duoc truyen vao
        if not (offsets is None):
            self.offsets = offsets
        else:
            # Tao ra mot mang chua so luong None bang so luong lop predictor layer
            self.offsets = [None] * predictor_sizes.shape[0]

        # bool cho phep dua cac bbx nam dung trong ranh gioi cua hinh anh
        self.clip_boxes = clip_boxes
        # mang variances giup chuan hoa cac toa do cua bbx
        self.variances = variances
        # option cho phep chi nhan mot bbx co IoU cao nhat voi ground truth hoac lay
        # nhieu hon cac bbx co nguong IoU cao hon mot nguong nao do
        self.matching_type = matching_type
        # iou cho cac positive
        self.pos_iou_threshold = pos_iou_threshold
        # ioi cho cac negative
        self.neg_iou_limit = neg_iou_limit
        self.border_pixels = border_pixels
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        # Tinh so luong anchor box tren moi vi tri khong gian cho moi lop predictor layers
        # Vi du: neu mot lop predictor co ba ty le khung hinh khac nhau la [1.0, 0.5, 2.0]
        # va duoc cho la du doan hai box co kich thuoc hoi khac nhau cho ty le khung hinh
        # ratios = 1.0 thi lop du doan do du doan tong cong bon box o moi ko gian vi tri
        # tren ban do tinh nang

        # Neu ratios cho tung lop duoc truyen vao
        if not (aspect_ratios_per_layer is None):
            # mang quan ly so luong cac box
            self.n_boxes = []
            # Lap qua cac danh sach ratios cho tung lop predictor layers
            for aspect_ratios in aspect_ratios_per_layer:
                # neu 1 nam trong aspect ratios va co 2 box duoc tao ra tu ratios = 1
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    # so luong box bang so luong cac ratios + 1
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    # Neu ko thi so luong box chi bang so luong cac ratios cho tung
                    # predictor layers
                    self.n_boxes.append(len(aspect_ratios))
        else:
            # Doi voi truong hop ratios global thi van tinh mot cach tuong tu
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

        ##################################################################################
        # Tinh toan cac Anchor box cho moi predictor layers
        ##################################################################################

        # Tinh toan cac anchor box cho moi lop predictor. Chung ta phai thuc hien viec
        # nay mot lan vi cac anchor box chi phu thuoc vao cau hinh model, ko phu
        # thuoc vao du lieu dau vao. Doi voi moi predictor layer (tuc la doi voi moi he
        # so ty le trong scales), cac tensor cho cac lop anchor box se co hinh dang la
        # `(feature_map_height, feature_map_width, n_boxes, 4)`.

        # Dieu nay luu tru cac anchor box cho moi lop predictor
        self.boxes_list = []

        # Cac danh sach sau day chi luu tru thong tin chan doan. Doi khi that huu ich khi co
        # cac diem trung tam, chieu cao, chieu rong, ... , cua cac box trong list
        
        # Chieu rong va chieu cao cho cac box cua moi lop predictor layer
        self.wh_list_diag = []

        # Khoang cach ngang doc giua hai hop bat ky cho moi lop predictor
        self.steps_diag = []

        # Offsets cho moi lop predictor
        self.offsets_diag = []

        # Cac center points voi dinh dang la (cx, cy) cho moi lop predictor layers
        self.centers_diag = []

        # lap qua tat ca cac lop predictor va tinh toan cac anchor box cho moi layer
        for i in range(len(self.predictor_sizes)):

            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                aspect_ratios=self.aspect_ratios[i],
                this_scale=self.scales[i],
                next_scale=self.scales[i+1],
                this_steps=self.steps[i],
                this_offsets=self.offsets[i],
                diagnostics=True)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

    def __call__(self, ground_truth_labels, diagnostics=False):
        '''
        Converts ground truth bounding box data into a suitable format to train an SSD model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, ymin, xmax, ymax)` (i.e. the 'corners' coordinate format), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        '''

        # Mapping to define which indices represent which coordinates in the ground truth.
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        ##################################################################################
        # Generate the template for y_encoded.
        ##################################################################################

        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)

        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################

        # Match the ground truth boxes to the anchor boxes. Every anchor box that does not have
        # a ground truth match and for which the maximal IoU overlap with any ground truth box is less
        # than or equal to `neg_iou_limit` will be a negative (background) box.

        y_encoded[:, :, self.background_id] = 1 # All boxes are background boxes by default.
        n_boxes = y_encoded.shape[1] # The total number of boxes that the model predicts per batch item
        class_vectors = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(batch_size): # For each batch item...

            if ground_truth_labels[i].size == 0: continue # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:,[xmax]] - labels[:,[xmin]] <= 0) or np.any(labels[:,[ymax]] - labels[:,[ymin]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                         "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. Degenerate ground truth " +
                                         "bounding boxes will lead to NaN errors during the training.")

            # Maybe normalize the box coordinates.
            if self.normalize_coords:
                labels[:,[ymin,ymax]] /= self.img_height # Normalize ymin and ymax relative to the image height
                labels[:,[xmin,xmax]] /= self.img_width # Normalize xmin and xmax relative to the image width

            # Maybe convert the box coordinate format.
            if self.coords == 'centroids':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2centroids', border_pixels=self.border_pixels)
            elif self.coords == 'minmax':
                labels = convert_coordinates(labels, start_index=xmin, conversion='corners2minmax')

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)] # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin,ymin,xmax,ymax]]], axis=-1) # The one-hot version of the labels for this batch item

            # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = iou(labels[:,[xmin,ymin,xmax,ymax]], y_encoded[i,:,-12:-8], coords=self.coords, mode='outer_product', border_pixels=self.border_pixels)

            # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
            #        This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
            #         ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
            #         such ground truth box.

            if self.matching_type == 'multi':

                # Get all matches that satisfy the IoU threshold.
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

                # Write the ground truth data to the matched anchor boxes.
                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, matches[1]] = 0

            # Third: Now after the matching is done, all negative (background) anchor boxes that have
            #        an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
            #        i.e. they will no longer be background boxes. These anchors are "too close" to a
            #        ground truth box to be valid background boxes.

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################

        if self.coords == 'centroids':
            y_encoded[:,:,[-12,-11]] -= y_encoded[:,:,[-8,-7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:,:,[-12,-11]] /= y_encoded[:,:,[-6,-5]] * y_encoded[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:,:,[-10,-9]] /= y_encoded[:,:,[-6,-5]] # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encoded[:,:,[-2,-1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        elif self.coords == 'corners':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-10]] /= np.expand_dims(y_encoded[:,:,-6] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-11,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-7], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
        elif self.coords == 'minmax':
            y_encoded[:,:,-12:-8] -= y_encoded[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-11]] /= np.expand_dims(y_encoded[:,:,-7] - y_encoded[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-10,-9]] /= np.expand_dims(y_encoded[:,:,-5] - y_encoded[:,:,-6], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encoded[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,-12:-8] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        '''
        Tinh toan mot array cac vi tri va kich thuoc ko gian cua cac anchor box cho 1 lop
        predictor layer co kich thuoc `feature_map_size == [feature_map_height, feature_map_width]`.

        Cac doi so:
            * feature_map_size (tuple):
                - Mot list hoac tuple `[feature_map_height, feature_map_width]`
                  voi cac kich thuoc ko gian ko gian ban do tinh nang de tao cac anchor box
            * aspect_ratios (list):
                - Mot danh sach cac so float, chua cac ty le khung hinh ratios ma cac anchor
                  box se duoc tao. Tat ca cac phan tu nay phai mang gia tri duy nhat.
            * this_scale (float):
                - Mot so float trong [0, 1], day la he so ty le hien tai de tao ra anchor box
            * next_scale (float): 
                - Mot float trong [0, 1], he so ty le lon hon tiep theo. chi lien quan den
                  truong hop `self.two_boxes_for_ar1 == True`.
            * diagnostics (bool, optional): Neu true, cac dau ra bo sung sau se duoc tra ve:
                1) Mot danh sach cac toa do center point `x` va `y` for each spatial location cho tung vi tri ko gian.
                2) Mot danh sach gom`(width, height)` cho moi box tuong ung voi aspect ratios
                3) Mot tuple chua `(step_height, step_width)`
                4) Mot tuple chua `(offset_height, offset_width)`
                Thong tin nay co the huu ich de chi hieu trong mot vai con so, tuc la cai
                ma cac luoi anchor box duoc ta ra trong nhu the nao, nghia la cac box khac nhau
                lon nhu the nao, va phan bo khong gian cua chung day dac nhu the nao,
                de xem luoi cac box co bao phu duoc duoc cac doi tuong trong hinh anh dau
                vao mot cach thich hop hay khong.

        Returns:
            Mot tensor numpy 4D voi kich thuoc
            `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` trong do kich thuoc
            cuoi cung chua xmin, xmax, ymin, ymax cho moi anchor box trong moi o cua ban do 
            tinh nang.
        '''
        # Tinh chieu rong va chieu cao cua cac box cho tung ty le khung hinh aspect ratio

        # mat ngan hon cua hinh anh se duoc su dung de tinh toan w va h bang cach su dung
        # scale va aspect ratios

        # lay canh ngan hon tu hinh anh dau vao
        size = min(self.img_height, self.img_width)

        # Tinh toan chieu rong va chieu cao cho cac box cho tat ca cac aspect ratios
        wh_list = []

        # Lap qua cac ty le khung hinh trong danh sach aspect ratio
        for ar in aspect_ratios:

            # Neu aspect ratios = 1
            if (ar == 1):
                # Tinh toan cac anchor box voi aspect ratio = 1
                # bang cach su dung scale cua predictor layers hien tai nhan size
                # (chieu nho hon trong hai chieu rong va cao cua hinh anh dau vao)
                box_height = box_width = this_scale * size
                # them chieu rong va chieu cao cua box vao trong mang quan ly wh_list
                wh_list.append((box_width, box_height))
                
                # neu cho phep tao 2 anchor box voi aspect ratios = 1
                if self.two_boxes_for_ar1:
                    # Tinh toan mot phien ban lon hon mot chut bang cach su dung
                    # gia tri trung binh hinh hoc cua gia tri scale cua lop predict
                    # layer hien tai nhan voi scale cua lop predictor layer tiep theo
                    # nhan voi canh nho hon (trong hai canh rong va cao) cua hinh anh dau vao
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    # them chieu rong va chieu cao cua box vao trong mang quan ly wh_list
                    wh_list.append((box_width, box_height))
            else:
                # Neu aspect ratios ko bang gia tri 1 thi tinh mot cach binh thuong chieu cao
                # va chieu rong cua cac anchor box
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))

        # Chuyen wh_list ve dang numpy array
        wh_list = np.array(wh_list)
        # So luong cac box
        n_boxes = len(wh_list)

        # Tinh toan luoi cua cac box center points. Chung giong het nhau cho tat ca cac
        # ty le khung hinh

        # Tinh toan step size, tuc la cac diem center cua cac anchor box cach nhau bao xa
        # va theo chieu ngang
        if (this_steps is None):
            # step theo chieu doc
            step_height = self.img_height / feature_map_size[0]
            # step theo chieu ngang
            step_width = self.img_width / feature_map_size[1]
        else:
            # neu this_steps co chua list hoac tuple va kich thuoc cua this_steps la 2
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                # step theo chieu doc la this_step[0]
                step_height = this_steps[0]
                # step theo chieu ngang la this_step[1]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        
        # Tinh toan cac do lech (offset), tuc la o cac gia tri pixel nao, diem trung tam
        # cua anchor box dau tien se o tren cung va tu ben trai cua hinh anh

        # neu this_offsets la None
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets

        # bay h chung ta da co offset va step sizes, tinh toan luoi toa do cac center point cua
        # cac anchor box
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        # vi du cx=5, cy=6 => np.meshgrid(cx, cy): [array([[5]]), array([[6]])]
        cx_grid, cy_grid = np.meshgrid(cx, cy)

        # Điều này là cần thiết để np.tile () thực hiện những gì chúng tôi muốn tiếp tục
        cx_grid = np.expand_dims(cx_grid, -1)
        # Điều này là cần thiết để np.tile () thực hiện những gì chúng tôi muốn tiếp tục
        cy_grid = np.expand_dims(cy_grid, -1)

        # Tao mot template tensor 4D co hinh dang `(feature_map_height, feature_map_width, n_boxes, 4)`
        # trong do kich thuoc cuoi cung se chua `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Thiet lap cho cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Thiet lap cho cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Thiet lap cho w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Thiet lap cho h

        # chuyen doi tu dinh dang `(cx, cy, w, h)` thanh dinh dang `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # Neu clip_boxes duoc bat hay cat toa do de cho cac bbx nam trong ranh gioi cua hinh anh
        if self.clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        # Neu normalize_coords duoc bat thi hay chuan hoa toa do cua cac bbx nam trong doan [0, 1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # TODO: Trien khai cac hop gioi han truc tiep cho cx, cy, w, h de chung ta ko phai chuyen
        # doi qua lai mot cach khong can thiet
        if self.coords == 'centroids':
            # Chuyen doi `(xmin, ymin, xmax, ymax)` ve dang `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif self.coords == 'minmax':
            # Chuyen doi dinh dang `(xmin, ymin, xmax, ymax)` ve dang `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the SSD model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes and the 4 variance values.
        '''
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
            # order of the tensor content will be identical to the order obtained from the reshaping operation
            # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encoding_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass