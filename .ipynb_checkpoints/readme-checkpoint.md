# Tính toán và phân tích điểm thi (Test Grade Calculator)
Chương trình được dùng để tính toán điểm thi cho nhiều lớp với sĩ số hàng nghìn học sinh.
Mục đích của chương trình giúp giảm thời gian chấm điểm.
Trong chương trình có áp dụng các functions (hàm) khác nhau trong Python để xử lý các tác vụ sau: 

* Mở các tập tin văn bản bên ngoài được yêu cầu với exception-handling
* Quét từng dòng của câu trả lời bài thi để tìm dữ liệu hợp lệ và cung cấp báo cáo tương ứng
* Chấm điểm từng bài thi dựa trên tiêu chí đánh giá (rubric) được cung cấp và báo cáo
* Tạo tập tin kết quả được đặt tên thích hợp

## Cài đặt
Chương trình được viết bằng ngôn ngữ python trên Jupyter Notebook:
<code>https://jupyter.org/</code>

## Cách dùng
Chương trình đã viết các hàm, người dùng chỉ cần chạy các hàm này để sử dụng:
* openClass(): mở file, nếu file không tồn tại sẽ báo 'File cannot found', nếu file được mở thì sẽ có thông báo 'Successfully opened <tên-file>.txt',
file đã mở rồi nếu nhập lại sẽ báo 'This file has already been opened.'
* analyzeClass(): mở file, phân tích file, nếu không có lỗi sẽ báo 'No error found', nếu có lỗi sẽ trả về lỗi gì và dòng bị lỗi.
* gradeClass(): mở file, như analyzeClass(), có thêm phần chấm điểm của học sinh, 
trả về các thông số thông kê: trung bình, điểm thấp nhất, điểm cao nhất, trung vị, khoảng điểm.
* outputGradeClass(): mở file, chấm điểm rồi trả về các file chứa điểm của từng học sinh trong lớp, được tách thành các file theo từng lớp.

