# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Ở `temperature=0.0`, phản hồi thường ổn định, tập trung vào một sự thật phổ biến và ít thay đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, model có xu hướng dùng cách diễn đạt đa dạng hơn, đưa thêm chi tiết hoặc ví dụ mới; ở 1.5, câu trả lời có thể sáng tạo hơn nhưng cũng dễ lan man, lặp ý hoặc xuất hiện chi tiết kém chắc chắn. Temperature không làm model có thêm kiến thức, mà thay đổi mức độ ngẫu nhiên khi chọn token tiếp theo.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Mình sẽ bắt đầu với `temperature=0.2` hoặc `0.3`. Chatbot hỗ trợ khách hàng cần trả lời nhất quán, chính xác, đúng chính sách và không tự ý sáng tạo thông tin; mức thấp giúp giảm sự dao động giữa các lần trả lời cùng một câu hỏi. Nếu sản phẩm cần giọng văn thân thiện hơn, có thể thử tăng lên khoảng `0.5`, nhưng vẫn phải kiểm thử các tình huống nhạy cảm và kết hợp prompt, dữ liệu tra cứu hoặc quy trình chuyển cho nhân viên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
Tổng output là `10.000 x 3 x 350 = 10.500.000` token, tương đương 10.500 đơn vị 1K token. Theo bảng giá trong `template.py`, GPT-4o tốn khoảng `10.500 x 0,010 = 105 USD`, còn GPT-4o-mini tốn `10.500 x 0,0006 = 6,30 USD`, nên GPT-4o đắt khoảng `105 / 6,30 = 16,67 lần` cho phần output. GPT-4o xứng đáng khi xử lý yêu cầu phức tạp, cần suy luận tốt hoặc tạo nội dung quan trọng như phân tích pháp lý/kỹ thuật có người kiểm duyệt; mini phù hợp cho FAQ, phân loại yêu cầu, tóm tắt ngắn và các tác vụ lặp lại với lưu lượng lớn. Đây là ước tính chỉ tính output, chưa tính input token và các chi phí khác.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
Với persona giáo viên tiểu học, phản hồi thường ngắn và dễ hiểu, dùng từ vựng đời thường, phép so sánh hoặc ví dụ gần gũi như các khối hộp và cuốn sổ ghi chép. Với persona chuyên gia tài chính, phản hồi thường dài và chính xác hơn về thuật ngữ, có thể nhắc đến sổ cái phân tán, cơ chế đồng thuận, hash và tính bất biến của dữ liệu. System prompt được gửi với role `system` trước user prompt trong `chat_with_system_prompt`, nên nó định hướng vai trò, mức độ chi tiết, ngôn ngữ và cách giải thích của model. Nó là chỉ dẫn hành vi chứ không tự bảo đảm thông tin luôn đúng, vì vậy vẫn cần kiểm tra nội dung đầu ra.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
Mình dùng đoạn 96 từ: “Việt Nam là một đất nước có nền văn hóa phong phú, lịch sử lâu đời và cảnh quan thiên nhiên đa dạng. Từ những cánh đồng lúa ở đồng bằng, bờ biển trải dài đến các dãy núi phía Bắc, mỗi vùng đều có nét đặc trưng riêng. Con người Việt Nam thường được biết đến với sự thân thiện, tinh thần hiếu khách và khả năng thích nghi tốt. Ẩm thực Việt Nam cũng rất nổi tiếng nhờ sự cân bằng giữa các hương vị chua, cay, mặn, ngọt và nhiều loại rau thơm.” Với `count_tokens` trong `template.py`, đoạn này có 125 token; công thức `số từ / 0.75` cho `96 / 0.75 = 128` token. Chênh lệch là 3 token, khoảng `3 / 128 x 100 = 2,34%` nếu lấy công thức ước lượng làm mốc. Tiếng Việt có nhiều dấu và cách tách âm tiết khiến tokenizer có thể phải chia chuỗi thành nhiều token nhỏ; vì vậy số token không tỷ lệ đơn giản với số từ như trong tiếng Anh, dù mức chênh lệch cụ thể còn phụ thuộc đoạn văn và tokenizer.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming quan trọng khi người dùng phải chờ một phản hồi dài, chẳng hạn chat trực tiếp, viết nội dung hoặc hỗ trợ kỹ thuật, vì ứng dụng có thể in từng chunk ngay khi model sinh ra và tạo cảm giác phản hồi nhanh hơn, dù tổng thời gian xử lý không nhất thiết giảm. Trong `template.py`, code lặp qua stream, dùng `chunk.choices[0].delta.content or ""` để tránh lỗi ở chunk cuối có nội dung `None`, đồng thời ghép các chunk thành `reply` để lưu history. Non-streaming phù hợp với câu trả lời ngắn, tác vụ backend cần nhận một kết quả hoàn chỉnh để parse, hoặc khi việc hiển thị từng phần làm giao diện phức tạp hơn lợi ích đem lại.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
Exponential backoff làm thời gian chờ tăng theo cấp số nhân, trong code là `base_delay * 2**attempt`, nên các lần retry sau có thêm thời gian để API giảm tải. Cách này giảm việc gửi request liên tục vào một dịch vụ đang quá tải và giúp cơ hội thành công tăng dần, trong khi vẫn phản hồi nhanh ở lần thử lại đầu tiên. Nếu hàng nghìn client đều retry cố định sau đúng 1 giây, chúng sẽ tạo ra các đợt request đồng thời; đợt retry lại làm hệ thống quá tải hơn, gây lỗi dây chuyền và có thể tạo vòng lặp “thundering herd”. Trong hệ thống thực tế nên thêm jitter ngẫu nhiên để các client không retry cùng một thời điểm.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
Persona mình chọn là trợ giảng lập trình thân thiện: “Bạn là trợ giảng thân thiện của khóa AI. Hãy giải thích khái niệm LLM và Python bằng tiếng Việt rõ ràng, ưu tiên ví dụ thực tế, trả lời ngắn gọn theo từng bước và nói rõ khi không chắc chắn thay vì bịa thông tin.” Từ “thân thiện” giúp giọng điệu dễ tiếp nhận, còn “bằng tiếng Việt” bảo đảm câu trả lời phù hợp với người học trong khóa. Cụm “theo từng bước” làm các hướng dẫn dễ làm theo; “nói rõ khi không chắc chắn” giúp hạn chế việc model trình bày phỏng đoán như sự thật. Persona này có thể truyền vào `run_assistant`, nơi nó được gửi trong message có role `system` ở mỗi lượt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
Hạn chế lớn nhất là history chỉ giữ tối đa 3 lượt, tức 6 message, nên trợ lý có thể quên yêu cầu hoặc quyết định được nói từ sớm; `run_assistant` thực hiện giới hạn này bằng `history = history[-6:]`. Cải thiện cụ thể là bổ sung bộ nhớ dài hạn dạng tóm tắt: trước khi cắt history, dùng một lời gọi model riêng hoặc hàm tóm tắt để cập nhật `conversation_summary`, sau đó gửi summary cùng system prompt và history gần nhất trong mỗi request. Có thể lưu summary theo người dùng trong SQLite hoặc một kho dữ liệu, giới hạn kích thước bằng token và chỉ cập nhật khi hội thoại vượt ngưỡng. Khi đó vẫn kiểm soát được chi phí và context window nhưng giữ lại mục tiêu, sở thích và các quyết định quan trọng của những lượt cũ.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
