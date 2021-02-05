# Mạng CNN nhận diện đeo khẩu trang sử dụng Tensorflow 

Đại dịch COVID-19 đã khiến cuộc sống của mọi người trở nên đảo lộn và gây ra rất nhiều khó khăn trong sinh hoạt như đi lại, ăn uống,.. Nhiều quốc gia đã bắt buộc người dân của họ phải đeo khẩu trang để bảo về sức khỏe của chính mình cũng như toàn thể xã hội. 
	 
Để bảo vệ mình khỏi đại dịch COVID-19 hầu hết mọi người trong chúng ta đều có xu hướng đeo khẩu trang. Chính vì thế mà việc kiểm tra những người trong đám đông có đeo khẩu trang ở các tụ điểm công cộng như sân bay, trung tâm mua sắm, công viên, .. hay không rất là cần thiết.
	 
Từ lý do đấy, tôi đã quyết định xây dựng một mô hình Mạng CNN đơn giản bằng việc sử dụng TensorFlow với thư viện Keras và OpenCV để phát hiện người có đang đeo khẩu trang hay không. Tôi mong ứng dụng của phương pháp này có thể rất hữu ích cho việc ngăn ngừa và kiểm soát COVID-19.

## Hướng tiếp cận
Việc phát hiện xem hình ảnh có người đeo khẩu trang hay không là một bài toán phân loại. Chúng ta phải phân loại các hình ảnh giữa 2 lớp rời rạc: Lớp có chứa khẩu trang và lớp không có.

## Dữ liệu
Để xây dựng mô hình này, tôi sẽ sử dụng tập dữ liệu về khẩu trang. Nó bao gồm khoảng 1.376 hình ảnh với 690 hình ảnh có người đeo khẩu trang và 686 hình ảnh có người không đeo khẩu trang.

## Cài đặt
Chạy file **face_mask_detector.ipynb**
Sử dụng jupyter notebook trên máy tính của bạn hoặc có thể sử dụng colab của google.

## Kiểm tra
Kiểm tra với dữ liệu vào là hình ảnh. Chạy file **predict_image.py**:

```python
python predict_image.py --image test_images/face_mask_test_1.png
```

Kiểm tra với dữ liệu vào là video. Chạy file **predict_video.py**:

```python
python predict_video.py
```
