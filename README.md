## SSD: Single-Shot MultiBox Detector implementation in Keras
---
### Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Examples](#examples)
4. [Dependencies](#dependencies)
5. [How to use it](#how-to-use-it)
6. [Download the convolutionalized VGG-16 weights](#download-the-convolutionalized-vgg-16-weights)
7. [Download the original trained model weights](#download-the-original-trained-model-weights)
8. [How to fine-tune one of the trained models on your own dataset](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
9. [ToDo](#todo)
10. [Important notes](#important-notes)
11. [Terminology](#terminology)

### Overview

Day la mot trien khai Keras cua kien truc SSD (https://arxiv.org/abs/1512.02325).

Kho luu tru hien cung cap cac kien truc mang sau:
* SSD300: [`keras_ssd300.py`](models/keras_ssd300.py)
* SSD512: [`keras_ssd512.py`](models/keras_ssd512.py)
* SSD7: [`keras_ssd7.py`](models/keras_ssd7.py) - một phiên bản 7 lớp nhỏ hơn có thể được huấn luyện từ đầu tương đối nhanh ngay cả trên GPU trung cấp, nhưng vẫn đủ khả năng cho các nhiệm vụ phat hien va thu nghiem cac doi tuong it phuc tap. Rõ ràng là bạn sẽ không nhận được kết quả hiện đại với kết quả đó, nhưng nó rất nhanh.

Neu ban muon sd mot trong nhung model da dc dao tao de thuc hien transfer learning (nghia la tinh chinh model da duoc dao tao tren bo du lieu cua rieng ban),  thi co mot [Jupyter notebook tutorial](weight_sampling_tutorial.ipynb) se giup ban lay dc cac mau con (sub-sample) trong so cac trong so da duoc dao tao de chung tuong thich voi tap du lieu cua ban.
Neu ban muon xay dung mot SSD voi kien truc mang co so cua minh, ban co the su dung [`keras_ssd7.py`](models/keras_ssd7.py) nhu la mot mau (template), no cung cap cac tai lieu va nhan xet giup ban.

### Performance

Duoi day la ket qua danh gia mAP cua cac trong so duoc dao tao trong keras, va ket qua danh gia cua mot mo hinh da duoc dao tao dua vao trien khai nay. Tat ca cac danh gia doi voi cac mo hinh deu duoc danh gia boi may chu thu nghiem Pascal VOC (for 2012 `test`) hoac tap lenh danh gia Pascal VOC Matlab (for 2007 `test`). Trong moi truong hop, ket qua khop hoac vuot mot chut so voi cac model Caffe ban dau. Tai cac lien ket den tat ca cac trong so da duoc dao tao bang keras co san duoi day:
<table width="70%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Do chinh xac trung binh (mAP)</td>
  </tr>
  <tr>
    <td>danh gia tren</td>
    <td colspan=2 align=center>VOC2007 test</td>
    <td align=center>VOC2012 test</td>
  </tr>
  <tr>
    <td>Dao tao tren<br>IoU rule</td>
    <td align=center width="25%">07+12<br>0.5</td>
    <td align=center width="25%">07+12+COCO<br>0.5</td>
    <td align=center width="25%">07++12+COCO<br>0.5</td>
  </tr>
  <tr>
    <td><b>SSD300</td>
    <td align=center><b>77.5</td>
    <td align=center><b>81.2</td>
    <td align=center><b>79.4</td>
  </tr>
  <tr>
    <td><b>SSD512</td>
    <td align=center><b>79.8</td>
    <td align=center><b>83.2</td>
    <td align=center><b>82.3</td>
  </tr>
</table>

Huấn luyện SSD300 từ đầu đến hội tụ trên Pascal VOC 2007 `trainval` va 2012 `trainval` tạo ra mAP tương tự trên Pascal VOC 2007 `test` như mẫu Caffe SSD300 "07 + 12" ban đầu. Bạn có thể tìm thấy một bản tóm tắt của đào tạo [here](training_summaries/ssd300_pascal_07+12_training_summary.md).

<table width="95%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Do chinh xac trung binh</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Mo hinh Caffe gocl</td>
    <td align=center>Cac trong dso duoc dao tao bang keras</td>
    <td align=center>Được đào tạo từ đầu</td>
  </tr>
  <tr>
    <td><b>SSD300 "07+12"</td>
    <td align=center width="26%"><b>0.772</td>
    <td align=center width="26%"><b>0.775</td>
    <td align=center width="26%"><b><a href="https://drive.google.com/file/d/1-MYYaZbIHNPtI2zzklgVBAjssbP06BeA/view">0.771</a></td>
  </tr>
</table>

### Examples

Dưới đây là một số ví dụ dự đoán về mẫu SSD300 "07 + 12" ban đầu được đào tạo đầy đủ (nghĩa là được đào tạo trên Pascal VOC2007 `trainval` va VOC2012 `trainval`). Dự đoán được đưa ra trên Pascal VOC2007 `test`.

| | |
|---|---|
| ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_05_no_gt.png) | ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_04_no_gt.png) |
| ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_01_no_gt.png) | ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_02_no_gt.png) |

Duoi day la mot so vi du du doan ve SSD7 (tuc phien ban 7 lop nho) duoc dao tao mot phan tren hai bo du lieu giao thong duong bo do [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) phat hanh voi tong so khoang 20,000 hinh anh chua 5 loai doi tuong (thong tin them trong [`ssd7_training.ipynb`](ssd7_training.ipynb)). Cac du doan ban thay duoi day dc dua ra sau 10.000 buoc dao tao o batch_size = 32. Phai thua nhan rang, o to la doi tuong tuong doi de phat hien va toi da chon mot vai vi du tot hon, nhung du sao mot mo hinh nho nhu vay co the lam duoc chi sau 10.000 lan lap di lap lai trong viec dao tao co the chap nhan duoc.
| | |
|---|---|
| ![img01](./examples/ssd7_udacity_traffic_pred_01.png) | ![img01](./examples/ssd7_udacity_traffic_pred_02.png) |
| ![img01](./examples/ssd7_udacity_traffic_pred_03.png) | ![img01](./examples/ssd7_udacity_traffic_pred_04.png) |

### Dependencies

* Python 3.x
* Numpy
* TensorFlow 1.x
* Keras 2.x
* OpenCV
* Beautiful Soup 4.x

Cac phu tro Theano va CNTK hien khong duoc ho tro

### How to use it

Kho lưu trữ này cung cấp các hướng dẫn về máy tính xách tay Jupyter để giải thích về đào tạo, suy luận và đánh giá, và có một loạt các giải thích trong các phần tiếp theo bổ sung cho sổ ghi chép.

Làm thế nào để sử dụng một mô hình được đào tạo để suy luận:
* [`ssd300_inference.ipynb`](ssd300_inference.ipynb)
* [`ssd512_inference.ipynb`](ssd512_inference.ipynb)

Lam the nao de train mot mo hinh (model):
* [`ssd300_training.ipynb`](ssd300_training.ipynb)
* [`ssd7_training.ipynb`](ssd7_training.ipynb)

Cách sử dụng một trong những mô hình được đào tạo được cung cấp để học chuyển (transfer learning) trên tập dữ liệu của riêng bạn:
* [Doc duoi day](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)

Lam the nao de danh gia mot mo hinh da duoc dao tao:
* Noi chung: [`ssd300_evaluation.ipynb`](ssd300_evaluation.ipynb)
* Tren MS COCO: [`ssd300_evaluation_COCO.ipynb`](ssd300_evaluation_COCO.ipynb)

Cach su dung trinh tao du lieu (data generator):
* Trình tạo dữ liệu được sử dụng ở đây có kho lưu trữ riêng với hướng dẫn chi tiết [here](https://github.com/pierluigiferrari/data_generator_object_detection_2d)

#### Training details

Các thiết lập đào tạo chung được đặt ra và giải thích trong [`ssd7_training.ipynb`](ssd7_training.ipynb) va trong [`ssd300_training.ipynb`](ssd300_training.ipynb). Hầu hết các thiết lập và giải thích đều giống nhau ở cả hai máy tính xách tay, vì vậy không quan trọng bạn nhìn vào cái nào để hiểu thiết lập đào tạo chung, nhưng các tham số trong [`ssd300_training.ipynb`](ssd300_training.ipynb) được cài đặt sẵn để sao chép thiết lập triển khai Caffe ban đầu để đào tạo về Pascal VOC, trong khi các tham số trong[`ssd7_training.ipynb`](ssd7_training.ipynb)được cài sẵn để đào tạo trên [Udacity traffic datasets](https://github.com/udacity/self-driving-car/tree/master/annotations).

Để đào tạo mô hình SSD300 gốc trên Pascal VOC:

1. Tải xuống bộ dữ liệu:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```
2. Tải về các trọng số cho mo hinh tich chap VGG-16 hoặc cho một trong các mô hình ban đầu được đào tạo được cung cấp dưới đây.
3. Đặt đường dẫn tệp cho bộ dữ liệu và trọng so mô hình tương ứng trong [`ssd300_training.ipynb`](ssd300_training.ipynb) và thực hiện các o (cells).

Tất nhiên, quy trình đào tạo SSD512 là giống nhau. Điều bắt buộc là bạn phải tải trọng so VGG-16 đã được đào tạo trước khi cố gắng đào tạo SSD300 hoặc SSD512 từ đầu, nếu không việc đào tạo có thể sẽ thất bại. Dưới đây là tóm tắt về đào tạo đầy đủ về mô hình SSD300 "07 + 12" để so sánh với đào tạo của riêng bạn:

* [SSD300 Pascal VOC "07+12" training summary](training_summaries/ssd300_pascal_07+12_training_summary.md)

#### Encoding and decoding boxes

Goi phu [`ssd_encoder_decoder`](ssd_encoder_decoder) chua tat ca cac ham va lop lien quan den encoding va decoding cac boxes. Encoding boxes co nghia la chuyen doi cac nhan that thanh dinh dang muc tieu ma ham loss function can trong qua trinh dao tao. Đây là quá trình mã hóa trong đó việc khớp các bbx dung với các hộp neo (anchor) (bai bao gọi chúng là các hộp mặc định và trong mã C ++ ban đầu, chúng được gọi là priors - tất cả đều giống nhau). Decoding boxes có nghĩa là chuyển đổi đầu ra mô hình thô trở lại định dạng nhãn đầu vào, đòi hỏi các quá trình chuyển đổi và lọc khác nhau, chẳng hạn như triệt tiêu không tối đa (NMS).

Để đào tạo mô hình, bạn cần tạo một thể hiện của `SSDInputEncoder` cần phải được chuyển đến bộ tạo dữ liệu (data generator). Trình tạo dữ liệu thực hiện phần còn lại, vì vậy bạn thường không cần gọi bất kỳ method thu cong nao trong `SSDInputEncoder`

Các mô hình có thể được tạo trong chế độ 'đào tạo' hoặc 'suy luận'. Trong chế độ 'đào tạo', mô hình đưa ra thang đo dự đoán thô vẫn cần được xử lý hậu kỳ với chuyển đổi tọa độ, ngưỡng tin cậy, triệt tiêu không tối đa, v.v. Cac function `decode_detections()` va `decode_detections_fast()` chịu trách nhiệm cho điều đó. Cái trước tuân theo việc triển khai Caffe ban đầu, đòi hỏi phải thực hiện NMS trên mỗi lớp đối tượng, trong khi cái sau thực hiện NMS trên toàn cầu trên tất cả các lớp đối tượng và do đó hiệu quả hơn, nhưng cũng hoạt động hơi khác. Đọc tài liệu để biết chi tiết về cả hai chức năng.Nếu một mô hình được tạo ở chế độ 'suy luận', lớp cuối cùng của nó là `DecodeDetections` layer, trong đó thực hiện tất cả các xử lý hậu kỳ`decode_detections()` duoc thuc hien, nhung trong TensorFlow. Điều đó có nghĩa là đầu ra của mô hình đã là đầu ra được xử lý sau. Để có thể huấn luyện, một mô hình phải được tạo trong chế độ 'đào tạo'. Các trọng số được đào tạo sau đó có thể được tải vào một mô hình được tạo ở chế độ 'suy luận'.

Một lưu ý về tọa độ bù hộp neo  (box offset) được mô hình sử dụng bên trong: Điều này có thể rõ ràng hoặc không rõ ràng đối với bạn, nhưng điều quan trọng là phải hiểu rằng mô hình không thể dự đoán tọa độ tuyệt đối cho các hộp giới hạn dự đoán. Để có thể dự đoán tọa độ hộp tuyệt đối, các lớp tich chập chịu trách nhiệm nội địa hóa sẽ cần tạo ra các giá trị đầu ra khác nhau cho cùng một đối tượng tại các vị trí khác nhau trong ảnh đầu vào. Tất nhiên, điều này là không thể: Đối với một đầu vào nhất định cho bộ lọc của lớp chập, bộ lọc sẽ tạo ra cùng một đầu ra bất kể vị trí không gian trong ảnh do các trọng số được chia sẻ. Đây là lý do tại sao mô hình dự đoán offset cho các hộp neo thay vì tọa độ tuyệt đối và tại sao trong quá trình đào tạo, tọa độ chân lý mặt đất tuyệt đối được chuyển đổi thành offset của hộp neo trong quá trình mã hóa. Thực tế là mô hình dự đoán độ lệch cho tọa độ hộp neo lần lượt là lý do tại sao mô hình chứa các lớp hộp neo không làm gì ngoài việc xuất tọa độ hộp neo để có thể bao gồm các thang đo đầu ra của mô hình. Nếu tenxơ đầu ra của mô hình không chứa tọa độ hộp neo, thông tin để chuyển đổi các độ lệch dự đoán trở lại tọa độ tuyệt đối sẽ bị thiếu trong đầu ra mô hình.

#### Using a different base network architecture

Nếu bạn muốn xây dựng một kiến trúc mạng cơ sở khác, bạn có thể sử dụng [`keras_ssd7.py`](models/keras_ssd7.py) nhu la mot template. Nó cung cấp tài liệu và comment để giúp bạn biến nó thành một mạng cơ sở khác. Kết hợp mạng cơ sở mà bạn muốn và thêm một lớp dự đoán lên trên mỗi lớp mạng mà bạn muốn đưa ra dự đoán. Tạo hai đầu dự đoán cho mỗi đầu, một để định vị, một để phân loại. Tạo một lớp hộp neo cho mỗi lớp dự đoán và đặt đầu ra của đầu địa phương hóa tương ứng làm đầu vào cho lớp hộp neo. Cấu trúc của tất cả các hoạt động định hình lại và nối ghép vẫn giữ nguyên, bạn chỉ cần đảm bảo bao gồm tất cả các lớp dự đoán và các lớp hộp neo của bạn.

### Download the convolutionalized VGG-16 weights

Để huấn luyện SSD300 hoặc SSD512 từ đầu, hãy tải xuống các trọng số của mo hinh tich chap VGG-16 hoàn toàn được đào tạo để hội tụ về phân loại ImageNet tại đây:

[`VGG_ILSVRC_16_layers_fc_reduced.h5`](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox).

Như với tất cả các tệp trọng so khác bên dưới, đây là cổng trực tiếp của tệp `.caffemodel` tương ứng được cung cấp trong kho lưu trữ của triển khai Caffe gốc.

### Download the original trained model weights

Dưới đây là các trọng số được chuyển cho tất cả các mô hình được đào tạo ban đầu. Tên tệp tương ứng với đối tác `.caffemodel` . Dấu hoa thị và chú thích đề cập đến những người trong README của [original Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd#models).

1. PASCAL VOC models:

    * 07+12: [SSD300*](https://drive.google.com/open?id=121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q), [SSD512*](https://drive.google.com/open?id=19NIa0baRCFYT3iRxQkOKCD7CpN6BFO8p)
    * 07++12: [SSD300*](https://drive.google.com/open?id=1M99knPZ4DpY9tI60iZqxXsAxX2bYWDvZ), [SSD512*](https://drive.google.com/open?id=18nFnqv9fG5Rh_fx6vUtOoQHOLySt4fEx)
    * COCO[1]: [SSD300*](https://drive.google.com/open?id=17G1J4zEpFwiOzgBmq886ci4P3YaIz8bY), [SSD512*](https://drive.google.com/open?id=1wGc368WyXSHZOv4iow2tri9LnB0vm9X-)
    * 07+12+COCO: [SSD300*](https://drive.google.com/open?id=1vtNI6kSnv7fkozl7WxyhGyReB6JvDM41), [SSD512*](https://drive.google.com/open?id=14mELuzm0OvXnwjb0mzAiG-Ake9_NP_LQ)
    * 07++12+COCO: [SSD300*](https://drive.google.com/open?id=1fyDDUcIOSjeiP08vl1WCndcFdtboFXua), [SSD512*](https://drive.google.com/open?id=1a-64b6y6xsQr5puUsHX_wxI1orQDercM)


2. COCO models:

    * trainval35k: [SSD300*](https://drive.google.com/open?id=1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj), [SSD512*](https://drive.google.com/open?id=1IJWZKmjkcFMlvaz2gYukzFx4d6mH3py5)


3. ILSVRC models:

    * trainval1: [SSD300*](https://drive.google.com/open?id=1VWkj1oQS2RUhyJXckx3OaDYs5fx2mMCq), [SSD500](https://drive.google.com/open?id=1LcBPsd9CJbuBw4KiSuE1o1fMA-Pz2Zvw)

### How to fine-tune one of the trained models on your own dataset

Nếu bạn muốn tinh chỉnh một trong những mô hình được đào tạo được cung cấp trên tập dữ liệu của riêng bạn, rất có thể tập dữ liệu của bạn không có cùng số lượng lớp với mô hình được đào tạo. Hướng dẫn sau đây giải thích cách giải quyết vấn đề này:

[`weight_sampling_tutorial.ipynb`](weight_sampling_tutorial.ipynb)

### ToDo

Những điều sau đây nằm trong danh sách việc cần làm, được xếp hạng theo mức độ ưu tiên. Đóng góp được hoan nghênh, nhưng vui lòng đọc [contributing guidelines](CONTRIBUTING.md).

1. Thêm định nghĩa mô hình và trọng so được đào tạo cho SSD dựa trên các mạng cơ sở khác như MobileNet, InceptionResNetV2 hoặc DenseNet.
2. Thêm hỗ trợ cho các phụ trợ Theano và CNTK. Yêu cầu chuyển các lớp tùy chỉnh và chức năng Loss Function từ TensorFlow sang phụ trợ Keras trừu tượng.

Hien dang lam:

* Mot Loss Function moi [Focal Loss](https://arxiv.org/abs/1708.02002).

### Important notes

* Tất cả các mô hình được đào tạo đã được đào tạo về trenMS COCO sử dụng các hệ số tỷ lệ hộp neo nhỏ hơn được cung cấp trong tất cả các máy tính xách tay Jupyter. Cụ thể, lưu ý rằng các mô hình '07 + 12 + COCO 'và '07 ++ 12 + COCO' sử dụng các hệ số tỷ lệ nhỏ hơn.

### Terminology

* "Anchor boxes": Bai bao goi la cac "default boxes", Trong cac trien khai C++ ban dau thi chung duoc goi la cac "prior boxes" hoac "priors", va trong bai bao ve Faster R-CNN thi goi chung la "anchor boxes". Tat ca cac thuat ngu nay deu chi den mot ky thuat, nhung toi thich cai ten "anchor boxes" boi vi toi tim duoc nhieu mo ta nhat ve thuat ngu nay. Toi goi chung la "prior boxes" hoac "priors" trong `keras_ssd300.py` va `keras_ssd512.py` để phù hợp với việc voi cac trien khai Caffe ban đầu, nhung o moi cho khac toi su dung ten goi la "anchor boxes" hoac "anchors".
* "Labels": Doi voi muc dich cua du an nay, bo du lieu bao gom "images" va "labels". Tat ca moi thong tin chu thich cua mot hinh anh thi duoc goi la "labels" cua hinh anh do: No ko chi bao gom ten cua doi tuong ma con bao gom ca toa do cac bbx cua hop gioi han. "Labels"  la cach goi ngan hon "annotations - chu thich". Toi cung su dung cac thuat ngu "labels" va "targets" it nieu thay the cho nhau trong suot tai lieu, mac du "targets" co nghia la nhan cu the trong boi canh dao tao.
* "Predictor layer": The "predictor layers" hoac  "predictors" là tất cả các lớp tich chập cuối cùng của mạng, tức là tất cả các lớp tich chập không cung cấp cho bất kỳ lớp chập tiếp theo nào.
