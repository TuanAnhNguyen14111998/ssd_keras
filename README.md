## SSD: Single-Shot MultiBox Detector trien khai trong Keras
---
### Noi dung

1. [Tong quan](#overview)
2. [Hieu suat](#performance)
3. [Vi du](#examples)
4. [Cac phu thuoc](#dependencies)
5. [Lan the nao de su dung no](#how-to-use-it)
6. [Tai ve trong so cua mang tich chap VGG16](#download-the-convolutionalized-vgg-16-weights)
7. [Tai ve trong so da dao tao cua model duoc dao tao ban dau](#download-the-original-trained-model-weights)
8. [Cach tinh chinh mot model da duoc dao tao tren bo du lieu cua chinh ban](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
9. [Cach lam](#todo)
10. [Cac ghi chu quan trong](#important-notes)
11. [Thuat ngu](#terminology)

### Overview

Đây là một cổng Keras của kiến truc mô hình SSD được giới thiệu bởi Wei Liu và cộng sự. trong bài báo [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

Mục tiêu chính của dự án này là tạo ra một triển khai SSD được ghi lại tốt cho những ai quan tâm đến sự hiểu biết ở mức độ thấp về mô hình. Các hướng dẫn, tài liệu và nhận xét chi tiết được cung cấp hy vọng sẽ giúp việc đào sâu vào mã dễ dàng hơn và điều chỉnh hoặc xây dựng theo mô hình so với hầu hết các triển khai khác ngoài đó (Keras hoặc cách khác) cung cấp rất ít tài liệu và nhận xét.

Kho lưu trữ hiện cung cấp các kiến trúc mạng sau:
* SSD300: [`keras_ssd300.py`](models/keras_ssd300.py)
* SSD512: [`keras_ssd512.py`](models/keras_ssd512.py)
* SSD7: [`keras_ssd7.py`](models/keras_ssd7.py) - một phiên bản 7 lớp nhỏ hơn có thể được huấn luyện từ đầu tương đối nhanh ngay cả trên GPU trung cấp, nhưng vẫn đủ khả năng cho các nhiệm vụ và thử nghiệm phát hiện đối tượng ít phức tạp hơn. Rõ ràng là bạn sẽ không nhận được kết quả hiện đại với kết quả đó, nhưng nó rất nhanh.

Nếu bạn muốn sử dụng một trong những mô hình được đào tạo được cung cấp để học chuyển giao (nghĩa là tinh chỉnh một trong những mô hình được đào tạo trên tập dữ liệu của riêng bạn), có một [hướng dẫn sổ ghi chép Jupyter](weight_sampling_tutorial.ipynb) giúp bạn lấy mẫu phụ các trọng số được đào tạo để chúng tương thích với tập dữ liệu của bạn, xem thêm bên dưới.

Nếu bạn muốn xây dựng một ổ SSD với kiến trúc mạng cơ sở của riêng mình, bạn có thể sử dụng [`keras_ssd7.py`](model/keras_ssd7.py) làm mẫu, nó cung cấp tài liệu và nhận xét để giúp bạn.

### Performance

Dưới đây là kết quả đánh giá mAP của các trọng số được chuyển và dưới đây là kết quả đánh giá của một mô hình được đào tạo từ đầu bằng cách sử dụng triển khai này. Tất cả các mô hình được đánh giá bằng máy chủ thử nghiệm Pascal VOC chính thức (cho năm 2012 `test`) hoặc tập lệnh đánh giá Pascal VOC Matlab chính thức (cho năm 2007` test`). Trong mọi trường hợp, kết quả khớp (hoặc vượt một chút) so với các mô hình Caffe ban đầu. Tải về các liên kết đến tất cả các trọng lượng được chuyển có sẵn dưới đây.

<table width="70%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Mean Average Precision</td>
  </tr>
  <tr>
    <td>evaluated on</td>
    <td colspan=2 align=center>VOC2007 test</td>
    <td align=center>VOC2012 test</td>
  </tr>
  <tr>
    <td>trained on<br>IoU rule</td>
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

Đào tạo một SSD300 từ đầu để hội tụ trên Pascal VOC 2007 `trainval` và 2012` trainval` tạo ra cùng một mAP trên Pascal VOC 2007` test` như mô hình Caffe SSD300 "07 + 12" ban đầu. Bạn có thể tìm thấy một bản tóm tắt của đào tạo [here](training_summaries/ssd300_pascal_07+12_training_summary.md).

<table width="95%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Mean Average Precision</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Original Caffe Model</td>
    <td align=center>Ported Weights</td>
    <td align=center>Trained from Scratch</td>
  </tr>
  <tr>
    <td><b>SSD300 "07+12"</td>
    <td align=center width="26%"><b>0.772</td>
    <td align=center width="26%"><b>0.775</td>
    <td align=center width="26%"><b><a href="https://drive.google.com/file/d/1-MYYaZbIHNPtI2zzklgVBAjssbP06BeA/view">0.771</a></td>
  </tr>
</table>

Các mô hình đạt được số khung hình trung bình mỗi giây (FPS) trên Pascal VOC trên điện thoại di động NVIDIA GeForce GTX 1070 (tức là phiên bản máy tính xách tay) và cuDNN v6. Có hai điều cần lưu ý ở đây. Đầu tiên, lưu ý rằng tốc độ dự đoán điểm chuẩn của việc triển khai Caffe ban đầu đã đạt được bằng cách sử dụng GPU TitanX và cuDNN v4. Thứ hai, bài báo nói rằng họ đã đo tốc độ dự đoán ở cỡ 8, mà tôi nghĩ không phải là cách đo tốc độ có ý nghĩa. Toàn bộ điểm đo tốc độ của một mô hình phát hiện là để biết có bao nhiêu hình ảnh liên tiếp riêng lẻ mà mô hình có thể xử lý mỗi giây, do đó đo tốc độ dự đoán trên các lô hình ảnh và sau đó suy ra thời gian dành cho mỗi hình ảnh riêng lẻ trong lô đó mục đích. Để dễ so sánh, bên dưới bạn tìm thấy tốc độ dự đoán cho việc triển khai Caffe SSD ban đầu và tốc độ dự đoán cho việc triển khai này trong cùng điều kiện, tức là ở cỡ lô 8. Ngoài ra, bạn tìm thấy tốc độ dự đoán cho việc triển khai này ở kích thước lô 1 , mà theo tôi là con số có ý nghĩa hơn.

<table width>
  <tr>
    <td></td>
    <td colspan=3 align=center>Frames per Second</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Original Caffe Implementation</td>
    <td colspan=2 align=center>This Implementation</td>
  </tr>
  <tr>
    <td width="14%">Batch Size</td>
    <td width="27%" align=center>8</td>
    <td width="27%" align=center>8</td>
    <td width="27%" align=center>1</td>
  </tr>
  <tr>
    <td><b>SSD300</td>
    <td align=center><b>46</td>
    <td align=center><b>49</td>
    <td align=center><b>39</td>
  </tr>
  <tr>
    <td><b>SSD512</td>
    <td align=center><b>19</td>
    <td align=center><b>25</td>
    <td align=center><b>20</td>
  </tr>
  <tr>
    <td><b>SSD7</td>
    <td align=center><b></td>
    <td align=center><b>216</td>
    <td align=center><b>127</td>
  </tr>
</table>

### Examples

Dưới đây là một số ví dụ dự đoán về mẫu SSD300 "07 + 12" được đào tạo đầy đủ (tức là được đào tạo trên Pascal VOC2007 `trainval` và VOC2012` trainval`). Các dự đoán đã được thực hiện trên Pascal VOC2007 `test`.

| | |
|---|---|
| ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_05_no_gt.png) | ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_04_no_gt.png) |
| ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_01_no_gt.png) | ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_02_no_gt.png) |

Dưới đây là một số ví dụ dự đoán về SSD7 (tức là phiên bản 7 lớp nhỏ) được đào tạo một phần về hai bộ dữ liệu giao thông đường bộ do [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) với khoảng 20.000 hình ảnh trong tổng số và 5 loại đối tượng (thông tin thêm trong [`ssd7_training.ipynb`](ssd7_training.ipynb)). Các dự đoán bạn thấy dưới đây được đưa ra sau 10.000 bước đào tạo ở cỡ lô 32. Phải thừa nhận rằng, ô tô là đối tượng tương đối dễ phát hiện và tôi đã chọn một vài ví dụ tốt hơn, nhưng dù sao một mô hình nhỏ như vậy có thể làm được chỉ sau 10.000 lặp đi lặp lại đào tạo.

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

Các phụ trợ Theano và CNTK hiện không được hỗ trợ.

Khả năng tương thích Python 2: Việc triển khai này dường như hoạt động với Python 2.7, nhưng tôi không cung cấp bất kỳ hỗ trợ nào cho nó. Đó là năm 2018 và không ai nên sử dụng Python 2 nữa.

### How to use it

Kho lưu trữ này cung cấp các hướng dẫn về máy tính xách tay Jupyter để giải thích về đào tạo, suy luận và đánh giá, và có một loạt các giải thích trong các phần tiếp theo bổ sung cho sổ ghi chép.

Làm thế nào để sử dụng một mô hình được đào tạo để suy luận:
* [`ssd300_inference.ipynb`](ssd300_inference.ipynb)
* [`ssd512_inference.ipynb`](ssd512_inference.ipynb)

Cách đào tạo mo hinh (model):
* [`ssd300_training.ipynb`](ssd300_training.ipynb)
* [`ssd7_training.ipynb`](ssd7_training.ipynb)

Cách sử dụng một trong những mô hình được đào tạo được cung cấp để học chuyển trên tập dữ liệu của riêng bạn:
* [Read below](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)

Làm thế nào để đánh giá một mô hình được đào tạo:
* In general: [`ssd300_evaluation.ipynb`](ssd300_evaluation.ipynb)
* On MS COCO: [`ssd300_evaluation_COCO.ipynb`](ssd300_evaluation_COCO.ipynb)

Cách sử dụng trình tạo dữ liệu:
* The data generator used here has its own repository with a detailed tutorial [here](https://github.com/pierluigiferrari/data_generator_object_detection_2d)

#### Training details

Các thiết lập đào tạo chung được đặt ra và giải thích trong [`ssd7_training.ipynb`](ssd7_training.ipynb) và trong[`ssd300_training.ipynb`](ssd300_training.ipynb). Hầu hết các thiết lập và giải thích đều giống nhau ở cả hai máy tính xách tay, vì vậy không quan trọng bạn phải nhìn vào thiết lập đào tạo chung nào, nhưng các tham số trong [`ssd300_training.ipynb`](ssd300_training.ipynb) được cài đặt sẵn để sao chép thiết lập triển khai Caffe ban đầu để đào tạo về Pascal VOC, trong khi các tham số trong [`ssd7_training.ipynb`](ssd7_training.ipynb) được cài sẵn để đào tạo trên [Udacity traffic datasets](https://github.com/udacity/self-driving-car/tree/master/annotations).

Để đào tạo mô hình SSD300 gốc trên Pascal VOC:

1. Tai tap datasets:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```
2. Tải về các trọng số cho VGG-16 tích chập hoặc cho một trong các mô hình ban đầu được đào tạo được cung cấp dưới đây.
3. Đặt đường dẫn tệp cho bộ dữ liệu và trọng lượng mô hình tương ứng trong [`ssd300_training.ipynb`](ssd300_training.ipynb) và thực thi các ô.

Tất nhiên, quy trình đào tạo SSD512 là giống nhau. Điều bắt buộc là bạn phải tải trọng lượng VGG-16 đã được đào tạo trước khi cố gắng đào tạo SSD300 hoặc SSD512 từ đầu, nếu không việc đào tạo có thể sẽ thất bại. Dưới đây là tóm tắt về đào tạo đầy đủ về mô hình SSD300 "07 + 12" để so sánh với đào tạo của riêng bạn:

* [SSD300 Pascal VOC "07+12" training summary](training_summaries/ssd300_pascal_07+12_training_summary.md)

#### Encoding and decoding boxes

The [`ssd_encoder_decoder`](ssd_encoder_decoder) gói phụ chứa tất cả các hàm và các lớp liên quan đến hộp mã hóa và giải mã. Hộp mã hóa có nghĩa là chuyển đổi nhãn sự thật mặt đất thành định dạng mục tiêu mà hàm mất mát cần trong quá trình đào tạo. Đây là quá trình mã hóa trong đó việc khớp các hộp sự thật với các hộp neo (tờ giấy gọi chúng là các hộp mặc định và trong mã C ++ ban đầu, chúng được gọi là linh mục - tất cả đều giống nhau). Các hộp giải mã có nghĩa là chuyển đổi đầu ra mô hình thô trở lại định dạng nhãn đầu vào, đòi hỏi nhiều quá trình chuyển đổi và lọc khác nhau như triệt tiêu không tối đa (NMS).

Để huấn luyện mô hình, bạn cần tạo một thể hiện của 'SSDInputEncoder` cần được truyền đến trình tạo dữ liệu. Trình tạo dữ liệu thực hiện phần còn lại, do đó bạn thường không cần gọi bất kỳ phương thức nào của 'SSDInputEncoder`.

Các mô hình có thể được tạo trong chế độ 'đào tạo' hoặc 'suy luận'. Trong chế độ 'đào tạo', mô hình đưa ra thang đo dự đoán thô vẫn cần được xử lý hậu kỳ với chuyển đổi tọa độ, ngưỡng tin cậy, triệt tiêu không tối đa, v.v. Các hàm `decode_detections ()` và `decode_detections_fast ()` cho điều đó Cái trước tuân theo việc triển khai Caffe ban đầu, đòi hỏi phải thực hiện NMS trên mỗi lớp đối tượng, trong khi cái sau thực hiện NMS trên toàn cầu trên tất cả các lớp đối tượng và do đó hiệu quả hơn, nhưng cũng hoạt động hơi khác. Đọc tài liệu để biết chi tiết về cả hai chức năng. Nếu một mô hình được tạo ở chế độ 'suy luận', thì lớp cuối cùng của nó là lớp `DecodeDetections`, thực hiện tất cả quá trình xử lý hậu kỳ mà` decode_detections () `thực hiện, nhưng trong TensorFlow. Điều đó có nghĩa là đầu ra của mô hình đã là đầu ra được xử lý sau. Để có thể huấn luyện, một mô hình phải được tạo trong chế độ 'đào tạo'. Các trọng số được đào tạo sau đó có thể được tải vào một mô hình được tạo ở chế độ 'suy luận'.

Một lưu ý về tọa độ bù hộp neo được mô hình sử dụng bên trong: Điều này có thể rõ ràng hoặc không rõ ràng đối với bạn, nhưng điều quan trọng là phải hiểu rằng mô hình không thể dự đoán tọa độ tuyệt đối cho các hộp giới hạn dự đoán. Để có thể dự đoán tọa độ hộp tuyệt đối, các lớp chập chịu trách nhiệm nội địa hóa sẽ cần tạo ra các giá trị đầu ra khác nhau cho cùng một đối tượng tại các vị trí khác nhau trong ảnh đầu vào. Tất nhiên, điều này là không thể: Đối với một đầu vào nhất định cho bộ lọc của lớp chập, bộ lọc sẽ tạo ra cùng một đầu ra bất kể vị trí không gian trong ảnh do các trọng số được chia sẻ. Đây là lý do tại sao mô hình dự đoán offset cho các hộp neo thay vì tọa độ tuyệt đối và tại sao trong quá trình đào tạo, tọa độ chân lý mặt đất tuyệt đối được chuyển đổi thành offset của hộp neo trong quá trình mã hóa. Thực tế là mô hình dự đoán độ lệch cho tọa độ hộp neo lần lượt là lý do tại sao mô hình chứa các lớp hộp neo không làm gì ngoài việc xuất tọa độ hộp neo để có thể bao gồm các thang đo đầu ra của mô hình. Nếu tenxơ đầu ra của mô hình không chứa tọa độ hộp neo, thông tin để chuyển đổi các độ lệch dự đoán trở lại tọa độ tuyệt đối sẽ bị thiếu trong đầu ra mô hình.

#### Using a different base network architecture

Nếu bạn muốn xây dựng một kiến trúc mạng cơ sở khác, bạn có thể sử dụng [`keras_ssd7.py`] (model / keras_ssd7.py) làm mẫu. Nó cung cấp tài liệu và ý kiến để giúp bạn biến nó thành một mạng cơ sở khác. Kết hợp mạng cơ sở mà bạn muốn và thêm một lớp dự đoán lên trên mỗi lớp mạng mà bạn muốn đưa ra dự đoán. Tạo hai đầu dự đoán cho mỗi đầu, một để định vị, một để phân loại. Tạo một lớp hộp neo cho mỗi lớp dự đoán và đặt đầu ra của đầu địa phương hóa tương ứng làm đầu vào cho lớp hộp neo. Cấu trúc của tất cả các hoạt động định hình lại và nối ghép vẫn giữ nguyên, bạn chỉ cần đảm bảo bao gồm tất cả các lớp dự đoán và các lớp hộp neo của bạn.

### Download the convolutionalized VGG-16 weights

Để huấn luyện SSD300 hoặc SSD512 từ đầu, hãy tải xuống các trọng số của mẫu VGG-16 được tích hợp hoàn toàn được đào tạo để hội tụ về phân loại ImageNet tại đây:

[`VGG_ILSVRC_16_layers_fc_reduced.h5`](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox).

Như với tất cả các tệp trọng lượng khác bên dưới, đây là cổng trực tiếp của tệp `.caffemodel` tương ứng được cung cấp trong kho lưu trữ của triển khai Caffe gốc.

### Download the original trained model weights

Dưới đây là các trọng số được chuyển cho tất cả các mô hình được đào tạo ban đầu. Tên tệp tương ứng với các đối tác `.caffemodel` tương ứng của chúng. Dấu hoa thị và chú thích đề cập đến những người trong README của [original Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd#models).

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

1. Thêm định nghĩa mô hình và trọng lượng được đào tạo cho SSD dựa trên các mạng cơ sở khác như MobileNet, InceptionResNetV2 hoặc DenseNet.
2. Thêm hỗ trợ cho các phụ trợ Theano và CNTK. Yêu cầu chuyển các lớp tùy chỉnh và chức năng mất từ TensorFlow sang phụ trợ Keras trừu tượng.

Hiện đang trong công trình:

* A new [Focal Loss](https://arxiv.org/abs/1708.02002) loss function.

### Important notes

* Tất cả các mô hình được đào tạo đã được đào tạo về MS COCO sử dụng các hệ số tỷ lệ hộp neo nhỏ hơn được cung cấp trong tất cả các máy tính xách tay Jupyter. Cụ thể, lưu ý rằng các mô hình '07 + 12 + COCO 'và '07 ++ 12 + COCO' sử dụng các hệ số tỷ lệ nhỏ hơn.

### Terminology

*"Hộp neo": Bài báo gọi chúng là "hộp mặc định", trong mã C ++ ban đầu, chúng được gọi là "hộp trước" hoặc "linh mục" và giấy Faster R-CNN gọi chúng là "hộp neo". Tất cả các thuật ngữ đều có nghĩa giống nhau, nhưng tôi hơi thích cái tên "hộp neo" bởi vì tôi thấy nó là mô tả nhất về những cái tên này. Tôi gọi chúng là "các hộp trước" hoặc "linh mục" trong `keras_ssd300.py` và` keras_ssd512.py` để phù hợp với triển khai Caffe ban đầu, nhưng ở mọi nơi khác tôi sử dụng tên" hộp neo "hoặc" neo ".
* "Nhãn": Đối với mục đích của dự án này, bộ dữ liệu bao gồm "hình ảnh" và "nhãn". Tất cả mọi thứ thuộc về chú thích của một hình ảnh nhất định là "nhãn" của hình ảnh đó: Không chỉ nhãn thể loại đối tượng, mà còn cả tọa độ hộp giới hạn. "Nhãn" chỉ ngắn hơn "chú thích". Tôi cũng sử dụng thuật ngữ "nhãn" và "mục tiêu" ít nhiều có thể thay thế cho nhau trong toàn bộ tài liệu, mặc dù "mục tiêu" có nghĩa là nhãn cụ thể trong bối cảnh đào tạo.
* "Lớp dự đoán": "Lớp dự đoán" hoặc "lớp dự đoán" là tất cả các lớp chập cuối cùng của mạng, tức là tất cả các lớp chập không cung cấp cho bất kỳ lớp chập tiếp theo nào.
