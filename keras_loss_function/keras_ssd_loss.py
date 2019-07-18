'''
Loss function Keras tuong thich voi model SSD. Hien tai chi ho tro backend Tensorflow
'''

from __future__ import division
import tensorflow as tf

class SSDLoss:
    '''
    The SSD loss, xem them trong bai bao https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Cac doi so:
            * neg_pos_ratio (int, optional):
                - ty le toi da ratio cua cac doi tuong tieu cuc (background) cho cac
                   ground truth boxes positive de dua vao ham loss. Tat nhien khong co 
                   hop background ground truth boxes nao trong thuc te, nhung y_true chua
                   cac anchor box duoc gan nhan voi cac class background.
                   Vi so luong cac box cho nen trong y_true thuong se vuot qua so luong cac
                   box positive, cho nen can phai can bang anh huong cua chung doi voi loss
                   function. Mac dinh la 3 theo paper.
            * n_neg_min (int, optional):
                - so luong toi thieu cac hop ground truth boxes negative de co the cho vao
                  loss function trong 1 batch. Dieu nay co the duoc su dung de dam bao
                  rang model hoc duoc tu mot so luong neg toi thieu trong cac batch trong do
                  co rat it hoac tham chi ko co gi ca so voi cac ground truth boxes positive.
                  Mac dinh no la 0 va neu duoc su dung, no duoc dat thanh cac gia tri tuong ung
                  voi kich thuoc batch duoc su dung cho dao tao.
            * alpha (float, optional): Mot yeu to de can nhac loss localization trong viec 
                  toan cua tong loss. Mac dinh la 1.0 theo paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Tinh toan loss smooth L1, nhin vao tai lieu de biet them chi tiet

        Arguments:
            y_true (nD tensor): Mot tensorflow co kich thuoc nD chieu chua du lieu labels true.
                trong ngu canh nay hinh dang cua no la  `(batch_size, #boxes, 4)`  va chua
                toa do cua bbx trong do kich thuoc cuoi duoc dung theo dinh dang 
                `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): Mot tensorflow co kich thuoc giong het voi y_true chua du lieu
                du doan. 

        Returns:
            Smooth loss L1, mot tensor nD-1 tensorflow. Trong truong hop nay, mot tensor 2D
            co hinh dang (batch, n_boxes_total)

        Lien ket:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Tinh toan loss sotfmax

        Cac doi so:
            y_true (nD tensor): Mot tensorflow cua bat ky hinh dang nao co chua du lieu labels
                true. Trong truong hop nay, tensor nay co hinh dang la(batch_size, #boxes, #classes)
                va bao gom ground truth bounding box cho cac class
            y_pred (nD tensor): Mot tensorflow co cau truc giong het y_true chua du lieu du doan
                trong boi canh nay, no la cac predicted bounding box categories.

        Tra ve:
            Ham loss softmax, mot tensor co kich thuoc nD-1. Trong truong hop nay no la mot tensor
            2D co kich thuoc (batch, n_boxes_total)
        '''
        # Hay chac chan rang y_pred khong chua bat ky so 0 nao vi no se pha vo gia tri cua logarit
        y_pred = tf.maximum(y_pred, 1e-15)
        # Tinh toan ham loss logarit cua softmax
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Tinh toan cac loss function cua phan predict SSD model de dua gan gia tri du doan
        den cac labels true

        Cac doi so:
            y_true (array): Mot mang numpy array co hinh dang `(batch_size, #boxes, #classes + 12)`
                trong do #boxes la tong so boxes ma model du doan tren moi hinh anh. 
                Hay can than de dam bao rang, chi muc cua moi box da cho trong y_true giong
                voi chi muc cua box tuong ung voi y_pred.
                Truc cuoi cung phai co do dai #classes + 12 va chua `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 muc tuy y]`.
                Tam muc cuoi cung cua truc cuoi cung khong duoc ham nay su dung va do do
                noi dung cua chung khong lien quan, chung chi ton tai sao cho y_true 
                co hinh dang giong nhu y_pred, trong do 4 muc cuoi cung cua truc chua cac toa do
                cua cac anchor box, can thiet trong qua trinh suy luan.
                Quan trong: cac box ma ban muon cost function bo qua can phai co mot vector
                one-hot  class vector chua so 0.
            y_pred (Keras tensor): Du doan cua model. Hinh dang giong het voi y_true tuc la
                `(batch_size, #boxes, #classes + 12)`. truc cuoi cung phai chua cac muc 
                trong dinh dang  `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 muc tuy y]


        Returns:
            Mot so vo huong, tong cac loss thoa man loss location va loss confidence (classification)
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
        # Output dtype: tf.int32, ghi nho rang, n_boxes trong ngu canh nay bieu thi tong so box
        # tren moi hinh anh, khong phai so luong box tren moi cell
        n_boxes = tf.shape(y_pred)[1]

        # 1: Tinh toan loss cho class va cho moi box predictor cho moi box

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # Output shape: (batch_size, n_boxes)

        # 2: Tinh toan cac loss classification cho cac muc tieu tich cuc va tieu cuc

        # Tao mat na (mask) cho cac class positive va negative trong labels
        # Tensor co kich thuoc shape (batch_size, n_boxes)
        negatives = y_true[:,:,0]
        # Tensor co kich thuoc shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1))

        # Dem so luong cac positive class (lop tu 1 den m) trong y_true den toan bo batch
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item  (Keras loss functions must output one scalar loss value PER batch item, rather than just one scalar for the entire batch, that's why we're not summing across all axes).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any).

        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False) # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: Compute the total loss.

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
