# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Quang Duy  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Map từng concept trong lecture vào code bạn vừa viết trong lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Sử dụng SentenceTransformer `all-MiniLM-L6-v2` để tạo embedding cho từng câu và cosine similarity để xác định điểm tách. Threshold mặc định 0.85 giúp nhóm các câu có nội dung gần nhau thay vì chỉ cắt theo số ký tự, nhờ đó bảo toàn ngữ nghĩa tốt hơn basic chunking. Trong pipeline chính, hierarchical chunking tạo ra 125 chunks từ 26 tài liệu. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 xử lý tốt các truy vấn có từ khóa hoặc số liệu chính xác, trong khi Dense Search với embedding hỗ trợ tìm kiếm theo ngữ nghĩa. RRF kết hợp thứ hạng của hai phương pháp bằng công thức `1/(k + rank + 1)` mà không cần chuẩn hóa trực tiếp score của BM25 và Dense Search. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Sử dụng `BAAI/bge-reranker-v2-m3` để đánh giá lại các candidate dựa trực tiếp trên cặp query-document và chọn top-3 context. Kết quả thực nghiệm cho thấy Context Precision tăng từ 0.9250 ở baseline lên 0.9458 ở Production, nhưng Context Recall giảm còn 0.7833. Điều này cho thấy reranking top-3 tăng precision nhưng có thể loại bỏ một số context cần thiết. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Sử dụng Faithfulness, Answer Relevancy, Context Precision và Context Recall để đánh giá pipeline. Production đạt Faithfulness = 0.6223, Answer Relevancy = 0.6435, Context Precision = 0.9458 và Context Recall = 0.7833. Việc đánh giá nhiều metric giúp phát hiện trade-off mà chỉ nhìn retrieval accuracy sẽ không thấy được. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Contextual Prepend bổ sung câu mô tả vị trí/chủ đề trước nội dung chunk. `_enrich_single_call()` tối ưu chi phí bằng cách tạo summary, hypothesis questions, context và metadata trong một lần gọi `gpt-4o-mini`. Trong lần chạy thực tế, 125 chunks được enrich trong khoảng 382.1 giây. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - Khi đọc một số tài liệu PDF, pipeline báo: `PDF scan ảnh, không có text layer (cần OCR).`
  - Khi kết nối Qdrant xuất hiện cảnh báo: `Failed to obtain server version. Unable to check client-server compatibility.`
  - Hugging Face Hub cảnh báo: `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`
  - Trong lần chạy ban đầu, OpenAI API key chưa được cấu hình nên các chức năng cần API phải sử dụng fallback.
  - Khi chạy RAGAS cho baseline xuất hiện thông báo: `Failed to parse output. Returning None.`

- **Nguyên nhân gốc rễ & Cách debug:**
  - Với PDF scan, nguyên nhân là tài liệu chỉ chứa hình ảnh và không có text layer nên loader hiện tại không thể trích xuất nội dung. Pipeline được thiết kế bỏ qua các tài liệu này thay vì dừng toàn bộ chương trình. Hướng cải tiến là bổ sung OCR trong bước ingestion.
  - Với Qdrant, client không lấy được thông tin version của server. Pipeline vẫn tiếp tục hoạt động và quá trình indexing hoàn thành thành công.
  - Với Hugging Face, model vẫn tải được nhưng request không có `HF_TOKEN`, vì vậy tốc độ và rate limit có thể bị hạn chế.
  - Với OpenAI API, tôi kiểm tra lại biến `OPENAI_API_KEY` trong `config.py`, cấu hình API key và xác nhận bằng lệnh kiểm tra trước khi chạy lại `python main.py`.
  - Với RAGAS, pipeline sử dụng `try/except` để tránh toàn bộ quá trình evaluation bị dừng khi một output không parse được.
  - Tôi debug từng module độc lập bằng `pytest tests/test_m1.py -v` đến `pytest tests/test_m5.py -v` trước khi chạy tích hợp toàn bộ pipeline. Cách này giúp xác định lỗi thuộc module nào thay vì debug toàn hệ thống cùng lúc.

- **Kiến thức còn thiếu & Cách khắc phục:**
  - Tôi cần tìm hiểu thêm về OCR để xử lý các tài liệu PDF scan trước khi chunking.
  - Tôi cần nghiên cứu sâu hơn cách lựa chọn `top_k` cho retrieval và reranking. Kết quả hiện tại cho thấy top-3 giúp Context Precision cao nhưng có thể làm Context Recall giảm.
  - Tôi cần tìm hiểu thêm cách tách `enriched_text` dùng cho retrieval và `original_text` dùng làm evidence cho LLM. Việc này có thể giúp hạn chế ảnh hưởng của nội dung do enrichment sinh ra tới Faithfulness.
  - Tôi sẽ tiếp tục sử dụng RAGAS và failure analysis để benchmark từng thay đổi thay vì giả định rằng thêm một kỹ thuật mới luôn giúp pipeline tốt hơn.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Hệ thống hỏi đáp tài liệu nội bộ

#### 1. Hiện trạng

- **Pipeline hiện tại:** Hệ thống nhận các tài liệu nội bộ, chia tài liệu thành chunks, tạo embedding, lưu vector và tìm các đoạn có nội dung gần với câu hỏi người dùng trước khi gửi context cho LLM tạo câu trả lời.
- **Vấn đề / Bottlenecks đang gặp:** Dense Search đơn lẻ có thể không tốt với từ khóa và số liệu chính xác; chunk nhỏ có thể mất context; scanned PDF chưa được xử lý; retrieval có thể lấy được tài liệu liên quan nhưng bỏ sót một phần thông tin cần thiết; LLM có nguy cơ hallucination nếu context không đầy đủ. Ngoài ra, enrichment bằng LLM làm tăng latency và chi phí.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Sử dụng kết hợp Hierarchical và Structure-aware Chunking. Hierarchical Chunking phù hợp khi cần giữ quan hệ giữa parent và child, còn Structure-aware phù hợp với tài liệu có heading rõ ràng vì có thể bảo toàn section và metadata.

2. **Search retrieval:** Sử dụng Hybrid Search gồm BM25 + Dense Search và Reciprocal Rank Fusion. BM25 giúp xử lý keyword, tên riêng và số liệu chính xác, còn Dense Search hỗ trợ các câu hỏi diễn đạt khác với văn bản gốc.

3. **Reranking:** Sử dụng Cross-Encoder `BAAI/bge-reranker-v2-m3`. Tôi sẽ benchmark nhiều giá trị `top_k`, đặc biệt top-3 và top-5, để tìm điểm cân bằng giữa Context Precision và Context Recall thay vì cố định top-3 cho mọi truy vấn.

4. **Evaluation:** Sử dụng 4 metric RAGAS gồm Faithfulness, Answer Relevancy, Context Precision và Context Recall. Mỗi phiên bản pipeline sẽ chạy trên cùng một test set để so sánh. Các câu có điểm thấp sẽ được đưa vào failure analysis để xác định lỗi nằm ở chunking, retrieval, reranking hay generation.

5. **Enrichment:** Áp dụng Contextual Prepend, HyQA và Metadata Extraction. `enriched_text` sẽ ưu tiên dùng cho indexing/retrieval, còn `original_text` được giữ lại để cung cấp evidence cho LLM. Combined single-call enrichment được sử dụng để giảm số lần gọi API.

#### 3. Timeline triển khai

- **Tuần 1:** Hoàn thiện ingestion pipeline, bổ sung OCR cho scanned PDF và triển khai Hierarchical/Structure-aware Chunking.
- **Tuần 2:** Xây dựng Hybrid Search BM25 + Dense + RRF, thêm Cross-Encoder reranking và benchmark các giá trị `top_k`.
- **Tuần 3:** Thêm Contextual Prepend, HyQA, Metadata Extraction và tách enriched text khỏi original evidence.
- **Tuần 4:** Xây dựng test set, chạy RAGAS, phân tích failure cases và tối ưu lại pipeline dựa trên kết quả định lượng.