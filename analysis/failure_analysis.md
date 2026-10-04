# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Quang Duy  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8421 | 0.6223 | -0.2198 |
| Answer Relevancy | 0.7207 | 0.6435 | -0.0772 |
| Context Precision | 0.9250 | 0.9458 | +0.0208 |
| Context Recall | 0.9250 | 0.7833 | -0.1417 |

## Bottom-5 Failures

### #1
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Trả lời chính xác chu kỳ đổi mật khẩu theo chính sách bảo mật trong tài liệu.
- **Got:** Câu trả lời không được context hỗ trợ đầy đủ; RAGAS đánh giá Faithfulness = 0.0.
- **Worst metric:** Faithfulness = 0.0, Average Score = 0.3333.
- **Error Tree:** Output sai/không được hỗ trợ → Context có thể thiếu thông tin chính xác → Query OK → Kiểm tra Retrieval/Reranking và Generation.
- **Root cause:** Câu hỏi đã rõ ràng nên lỗi nhiều khả năng không nằm ở query. Top-3 sau reranking có thể không giữ đúng chunk chứa quy định về chu kỳ đổi mật khẩu, hoặc LLM đã sinh thêm thông tin không có trong context.
- **Suggested fix:** Kiểm tra top-20 Hybrid Search và top-3 Cross-Encoder; đặt `temperature=0`; thắt chặt system prompt để chỉ cho phép trả lời từ context.

### #2
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Xác định đúng cấp/người phê duyệt đối với trường hợp nghỉ không lương 20 ngày.
- **Got:** Câu trả lời không được context hỗ trợ đầy đủ; Faithfulness = 0.0.
- **Worst metric:** Faithfulness = 0.0, Average Score = 0.4459.
- **Error Tree:** Output sai/không faithful → Context có thể chưa chứa đúng điều kiện 20 ngày → Query OK → Kiểm tra M2/M3 và Generation.
- **Root cause:** Có thể retrieval lấy được chính sách nghỉ phép nói chung nhưng top-3 chưa chứa đúng điều khoản kết hợp giữa số ngày nghỉ và cấp phê duyệt.
- **Suggested fix:** Tăng ưu tiên BM25 cho các từ khóa “20 ngày”, “không lương”, “phê duyệt”; giữ metadata section và kiểm tra lại kết quả Cross-Encoder.

### #3
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Xác định đúng cấp có thẩm quyền phê duyệt thiết bị ở mức giá 55 triệu.
- **Got:** Câu trả lời không được context hỗ trợ đầy đủ; Faithfulness = 0.0.
- **Worst metric:** Faithfulness = 0.0, Average Score = 0.4506.
- **Error Tree:** Output sai/không faithful → Context có thể thiếu đúng ngưỡng giá → Query OK → Kiểm tra Retrieval/Reranking.
- **Root cause:** Dense Search có thể tìm được các đoạn có chủ đề mua sắm tương tự nhưng không chứa đúng threshold 55 triệu. Việc chỉ giữ top-3 sau reranking cũng có thể loại bỏ đoạn chứa quy định chính xác.
- **Suggested fix:** Tăng vai trò BM25 đối với số tiền và từ khóa “phê duyệt”; kiểm tra candidate trước/sau reranking và cân nhắc tăng `top_k` từ 3 lên 5.

### #4
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Trả lời chính xác số ký tự tối thiểu theo chính sách mật khẩu.
- **Got:** Câu trả lời không được context hỗ trợ đầy đủ; Faithfulness = 0.0.
- **Worst metric:** Faithfulness = 0.0, Average Score = 0.5000.
- **Error Tree:** Output sai/không faithful → Context có thể lấy nhầm đoạn bảo mật → Query OK → Kiểm tra M2/M3 và Generation.
- **Root cause:** Đây là factual query chứa yêu cầu về một con số chính xác. Dense retrieval có thể ưu tiên đoạn tương đồng về chủ đề bảo mật nhưng không chứa đúng quy định về độ dài mật khẩu.
- **Suggested fix:** Ưu tiên BM25 cho factual query có số liệu; kiểm tra RRF và Cross-Encoder có giữ chunk chứa chính xác quy định hay không; đặt generation `temperature=0`.

### #5
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Xác định chính xác nhân viên thử việc có thuộc đối tượng được hưởng bảo hiểm sức khỏe PVI hay không.
- **Got:** Câu trả lời không được context hỗ trợ đầy đủ; Faithfulness = 0.0.
- **Worst metric:** Faithfulness = 0.0, Average Score = 0.5000.
- **Error Tree:** Output sai/không faithful → Context có thể chứa PVI chung nhưng thiếu điều kiện “nhân viên thử việc” → Query OK → Kiểm tra M2/M3.
- **Root cause:** Semantic retrieval có thể ưu tiên chính sách PVI chung thay vì đoạn quy định eligibility riêng cho nhân viên thử việc.
- **Suggested fix:** Tăng ưu tiên chunk chứa đồng thời “thử việc” và “PVI”; sử dụng metadata/category để lọc đúng chính sách nhân sự trước khi reranking.

## Case Study (cho presentation)

**Question chọn phân tích:**  
Bao lâu phải đổi mật khẩu một lần?

**Error Tree walkthrough:**
1. Output đúng? → Không đạt yêu cầu Faithfulness; RAGAS cho điểm 0.0.
2. Context đúng? → Cần kiểm tra top-3 thực tế; Context Recall tổng thể giảm từ 0.9250 xuống 0.7833 nên có khả năng relevant chunk bị loại khỏi context cuối.
3. Query rewrite OK? → Có. Câu hỏi ngắn, rõ ràng, chứa trực tiếp chủ đề “đổi mật khẩu”, nên không cần rewrite.
4. Fix ở bước: → Kiểm tra M2 Hybrid Search và đặc biệt M3 Cross-Encoder trước. Nếu relevant chunk có trong top-20 nhưng mất khỏi top-3 thì điều chỉnh reranking/top-k. Nếu context đã đúng nhưng answer vẫn sai thì sửa generation prompt và đặt `temperature=0`.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Log top-20 kết quả Hybrid Search và top-3 sau Cross-Encoder cho từng failure.
- Tăng thử `RERANK_TOP_K` từ 3 lên 5 và chạy lại RAGAS để kiểm tra Context Recall.
- Giữ `enriched_text` phục vụ retrieval nhưng dùng `original_text` làm evidence cho LLM generation.
- Đặt `temperature=0` và yêu cầu LLM không suy diễn ngoài context.
- Bổ sung OCR cho 2 PDF scan đang bị bỏ qua.
- Chạy lại cùng test set 20 câu và so sánh 4 RAGAS metrics với kết quả hiện tại.