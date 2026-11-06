# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? 
Record này nói về cả bức ảnh vì Checkpoint yolo11n-cls.pt xếp hạng các lớp ImageNet-1K cho toàn bộ ảnh
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? 
Danh sách lớp được xác định trước trong bộ phân loại
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
class_id là mã để máy tra cứu và đối chiếu nhãn. class_name là tên để người đọc dễ hiểu và kiểm tra. taxonomy_name cho biết mã và tên đó thuộc bộ lớp nào
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? 
Guideline cần xác định đang gán nhãn cho đối tượng chính hay cho bối cảnh của cả ảnh, rồi đưa ra thứ tự ưu tiên
6. Vì sao không được lấy `score` của model làm ground truth?
score là điểm model gán cho dự đoán của chính nó. Trong ảnh này, cab đứng đầu nhưng thứ hạng đó chưa xác nhận cab là nhãn đúng cho ảnh có nhiều loại xe

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=cup ; `score`=0.451108 ; `bbox_xyxy`=[143.52, 268.13, 172.93, 301.81] ; `bbox_width`=29.42 ; `bbox_height`=33.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Box nằm ở trên cái bàn trong bức ảnh, góc trên bên trái ở vị trí trên cái bàn và nằm ở phía trên bên trái chiếc cốc, góc dưới bên phải cũng ở vị trí mặt bàn trong bức ảnh và ở phía dưới bên phải chiếc cốc
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 
Ngưỡng 0.2 có 53 predictions còn ngưỡng 0.6 có 25.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Reviewer phải kiểm tra thêm các hộp được giữ lại.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Vẽ hình chữ nhật nhỏ nhất bao trọn phần nhìn thấy của đối tượng. Nếu đối tượng bị che hoặc cắt ở mép ảnh, chỉ bao phần còn nhìn thấy trong khung hình.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Ở mép trái ảnh, cạnh bàn, có một người chỉ lộ một phần cẳng tay và bàn tay, tôi sẽ hỏi lab coach xem phần tay này có đủ để gán nhãn không.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]  
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box cho biết vị trí và kích thước của khung chữ nhật bao quanh vật thể. Polygon mô tả đường viền chi tiết hơn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để phân biệt từng đối tượng được phát hiện trong ảnh. Hai người có hai instance_id riêng nhưng cùng class_id vì đều thuộc lớp person. 
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ mask sát đường viền phần nhìn thấy của đối tượng. Khi vật khác che khuất, dừng biên mask tại mép che và để phần khuất bên ngoài mask.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Chiếc bát trắng ở góc dưới bên trái che một phần mặt bàn. Mình sẽ áp dụng guideline vì có thể nhận ra đường viền của bát. Khi vẽ mask cho bàn, mình loại vùng bát đang che và chỉ tô phần mặt bàn nhìn thấy.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | là một nhãn lớp của mỗi ảnh | Ảnh traffic có nhiều loại xe. JSON xếp cab đứng đầu (0.510915), trong khi ảnh có nhiều xe buýt nổi bật. | Chọn một lớp trong taxonomy theo tiêu chí của guideline. | Kiểm tra lớp được chọn có đúng tiêu chí ưu tiên và thuộc taxonomy hay không. |
| Phát hiện vật thể | là nhãn của vật thể và đơn vị bounding box | ở mép trái của ảnh traffic có phần tay nhưng được gán nhãn là person, cần phải xác định xem vật thể có đủ điều kiện để gán nhãn hay không | Gán lớp và vẽ box sát phần nhìn thấy của từng vật thể đủ điều kiện. | Kiểm tra thiếu hoặc trùng box, sai lớp và box vẽ lệch. |
| Instance segmentation | nhãn và mask của vật thể | Trong PNG kitchen, chiếc bát trắng ở góc dưới bên trái che một phần mặt bàn. Biên mask của bàn cần dừng ở mép bát. | Vẽ mask theo đường viền phần nhìn thấy của từng vật thể. | Kiểm tra mask có lấn sang vật bên cạnh, bỏ sót phần nhìn thấy hoặc gộp hai vật thành một hay không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Tôi không chia sẻ dữ liệu được cung cấp trong bài cho người ngoài hoặc người không phận sự.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi tôi sẽ dừng và báo cho lab coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
