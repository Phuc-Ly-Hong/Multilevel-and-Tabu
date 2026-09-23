# Tune tham số Multilevel_Tabu_no_lwt bằng irace

Thư mục này chứa toàn bộ setup để dùng **irace** dò tham số tự động cho
`src/Multilevel_Tabu_no_lwt.cpp`.

## 1. Các thay đổi đã làm trong file C++

`main()` giờ nhận thêm các flag dạng `--ten_tham_so gia_tri` (thứ tự không
quan trọng), bên cạnh cách gọi cũ `exe <instance>`:

```
Multilevel_Tabu_no_lwt.exe <instance.txt> [--seed N]
    [--max_levels N] [--merge_ratio F]
    [--tabu_factor F] [--tabu_cap N]
    [--iter_k F]
    [--delta1 F] [--delta2 F] [--delta3 F] [--delta4 F]
```

Cuối chương trình, ngoài các log cũ, chương trình in thêm đúng một dòng:

```
IRACE_RESULT <giá_trị_mục_tiêu>
```

trong đó giá trị = `fitness` của lời giải tốt nhất, cộng thêm phạt `1e6` nếu
lời giải không khả thi (`is_feasible == false`). Đây là dòng mà script tune
đọc để biết cấu hình nào tốt hơn.

Các tham số/biến toàn cục mới thêm vào code (trước đây là hằng số cứng):

| Biến C++ | Trước đây | Bây giờ |
|---|---|---|
| `delta1..delta4` | `const double` cố định | biến, override qua `--delta1`..`--delta4` |
| `TABU_TENURE` | `min(ceil(n/4), 10)` | `min(ceil(n * TABU_TENURE_FACTOR), TABU_TENURE_CAP)`, tune qua `--tabu_factor`, `--tabu_cap` |
| `MAX_ITER` (số vòng lặp/1 level) | bậc thang cố định theo kích thước instance (dòng 189-220), tương đương K≈10 (500/50=1000/100=2000/200=10) | ghi đè bằng `--iter_k F`: `MAX_ITER = round(K × số_khách_hàng)`, tự scale theo size thay vì 1 số tuyệt đối cố định |
| seed RNG | luôn `time(nullptr)` | dùng `--seed` nếu có (bắt buộc để irace tái lập kết quả) |

`max_no_improve_segment` (dòng ~973, giá trị cố định = 8) **không đưa vào
tune** theo yêu cầu — vẫn giữ nguyên hằng số cứng như code gốc.

`alpha1` (hệ số phạt drone_violation trong fitness) **không đưa vào tune**
theo yêu cầu — vẫn giữ cố định = 1.0 như code gốc, không nhận flag CLI nữa.

`alpha2` và `Beta` (dòng ~100-101) hiện **không được dùng ở đâu trong code**
— tôi để nguyên, chưa đưa vào tune vì chúng chưa có tác dụng gì với fitness.

## 1b. Chốt cứng 3 tham số (không tune nữa)

Theo yêu cầu, `max_levels = 5`, `tabu_factor = 0.25`, `delta1 = 0.4` được
**chốt cứng** — không còn nằm trong `parameters.txt`, mà được truyền cố định
trong `FIXED_ARGS` ở đầu `tune.R` cho mọi lần chạy. Chỉ còn **6 tham số**
irace thật sự tune: `merge_ratio`, `tabu_cap`, `iter_k`, `delta2`, `delta3`,
`delta4`.

## 1c. Bước nhảy (step size) cho các tham số thực

irace không hỗ trợ khai báo step-size trực tiếp cho tham số kiểu `r` (thực)
trong `parameters.txt`, nên các tham số thực còn lại được biểu diễn dưới
dạng **số bước nguyên** (`merge_ratio_steps`, `iter_k_steps`,
`delta2_steps`, `delta3_steps`, `delta4_steps`) — `tune.R` tự nhân lại với
độ dài bước trước khi truyền cho exe (xem `STEP_SIZES` trong `tune.R`):

| Tham số | Bước nhảy | Miền số bước | Số giá trị khả dĩ |
|---|---|---|---|
| `merge_ratio` | 0.005 | 10 – 60 | 51 |
| `tabu_cap` | (nguyên sẵn, không đổi) | 5 – 30 | 26 |
| `iter_k` | 0.5 | 6 – 40 | 35 |
| `delta2` | 0.01 | 5 – 60 | 56 |
| `delta3` | 0.01 | 1 – 40 | 40 |
| `delta4` | 0.01 | 5 – 60 | 56 |

`iter_k` dùng bước 0.5 thay vì 0.01 như các delta — vì miền của nó (3–20,
biên độ 17) rộng hơn nhiều so với delta (biên độ ~0.4–0.55), nếu áp 0.01 sẽ
ra tới 1701 giá trị khác nhau, không hợp lý so với các tham số còn lại.

## 2. Cài đặt môi trường (chỉ cần làm 1 lần)

1. Cài **R**: tải tại https://cran.r-project.org/bin/windows/base/ (bản mới
   nhất), cài xong nhớ tick "Add R to PATH" hoặc tự thêm `Rscript.exe`
   (thường ở `C:\Program Files\R\R-x.x.x\bin`) vào PATH.
2. Mở PowerShell, cài gói irace:
   ```powershell
   Rscript -e "install.packages('irace', repos='https://cloud.r-project.org')"
   ```
3. Build lại executable với code đã sửa:
   ```powershell
   g++ -O2 -std=c++17 -o src/Multilevel_Tabu_no_lwt.exe src/Multilevel_Tabu_no_lwt.cpp
   ```
   > Lưu ý: khi tôi build thử trên máy này, linker (`ld.exe` của MSYS2/MinGW)
   > báo lỗi `ld returned 116 exit status` — kể cả với 1 file "hello world"
   > trống, nên đây là vấn đề môi trường (rất có thể Windows Defender / một
   > antivirus đang khoá file `.exe` mới tạo), không phải lỗi trong code.
   > Nếu bạn gặp lại lỗi này: thử build lại lần nữa (đôi khi transient), hoặc
   > thêm thư mục `D:\New folder` vào danh sách loại trừ của Windows Defender
   > (Windows Security → Virus & threat protection → Exclusions), rồi build
   > lại.

## 3. Chạy tuning

```powershell
cd tuning
Rscript tune.R
```

Mặc định `tune.R` dùng:
- **Instance train**: 36 file trong `tuning/train-instances/`:
  - toàn bộ `50.*.txt` (16 file) và `100.*.txt` (16 file)
  - `200.*.txt`: mỗi nhóm "grid" (`.10`, `.20`, `.30`, `.40`) lấy 1 replicate
    khác nhau cho đa dạng (`200.10.1`, `200.20.2`, `200.30.3`, `200.40.4`)
  - **Không có size 500** (bỏ theo yêu cầu — quá chậm, chưa có số liệu để
    ước lượng thời gian, để dành tune riêng sau nếu cần)
  Tách riêng khỏi bộ bạn dùng để báo cáo kết quả cuối cùng, tránh overfit
  tham số vào đúng bộ test.
- **Ngân sách**: `MAX_EXPERIMENTS = 25000` lần chạy thuật toán — chỉnh trong
  đầu file `tune.R` (khoá `MAX_EXPERIMENTS`). Nếu muốn test pipeline nhanh
  trước (kiểm tra không lỗi/không crash) thì tạm hạ xuống 500–1000, chạy
  thử, rồi trả lại 25000 cho lần tune "thật".
- **Song song**: `N_PARALLEL = 20` (đặt cho máy 24 nhân, chừa 4 nhân cho hệ
  điều hành/tác vụ khác) — irace tự dùng cluster kiểu PSOCK, chạy được trên
  Windows. Đổi lại nếu chạy trên máy khác có số nhân khác.
- **Timeout mỗi lần chạy**: `PER_RUN_TIMEOUT = 600` giây — nới rộng hơn mức
  mặc định vì có instance 200 khách hàng và chạy song song 20 tiến trình
  cùng lúc có thể làm mỗi lần chạy chậm hơn do tranh chấp CPU/bộ nhớ.

Kết quả:
- `tuning/elite-configurations.csv` — vài bộ tham số tốt nhất (mặc định
  irace trả ~5-7 bộ ngang nhau về thống kê).
- `tuning/irace-log.Rdata` — log đầy đủ, có thể mở lại bằng
  `irace::plotResults` hoặc phân tích thêm sau này.

## 4. Dùng kết quả

Sau khi có `elite-configurations.csv`, lấy bộ tham số hàng đầu (thường là
dòng đầu, `.ID. = 1` hoặc xếp theo cột thứ tự) rồi:

- **Cách nhanh**: gọi trực tiếp exe với các flag đó khi chạy thí nghiệm thật
  (ví dụ sửa `scripts/run_experiments.py` để truyền thêm các flag này khi
  gọi `Multilevel_Tabu_no_lwt.exe`).
- **Cách "chốt cứng"**: sửa giá trị mặc định trong code C++ (dòng khai báo
  `delta1..delta4`, `TABU_TENURE_FACTOR`, `TABU_TENURE_CAP`, `alpha1`,
  `MERGE_RATIO`, `MAX_LEVELS`) thành giá trị tune
  được, rồi build lại — để không cần truyền flag nữa trong các lần chạy sau.

## 5. Mở rộng thêm

- Muốn tune riêng theo từng nhóm kích thước bài toán (giống OSP paper tune
  riêng UC-1/UC-2/UC-3): tạo thêm thư mục instance train khác (vd
  `train-instances-large/`) và chạy `tune.R` riêng cho từng bộ, chỉnh
  `INSTANCES_DIR` ở đầu file.
- Muốn thêm tham số mới vào tune: (1) thêm biến global trong `.cpp`, (2)
  thêm nhánh `else if (key == "--ten_moi") ...` trong `main()`, (3) thêm
  dòng trong `parameters.txt`, (4) thêm dòng vào `switch_map` trong
  `tune.R`.
