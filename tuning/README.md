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

`max_levels = 5`, `tabu_factor = 0.25`, `delta1 = 0.8` được **chốt cứng** —
không nằm trong `parameters.txt`, truyền cố định trong `FIXED_ARGS` ở đầu
`tune.R`. Chỉ còn **5 tham số** irace thật sự tune: `merge_ratio`,
`tabu_cap`, `iter_k`, `delta2`, `delta3`.

## 1c. Bước nhảy (step size) cho các tham số thực

irace không hỗ trợ khai báo step-size trực tiếp cho tham số kiểu `r` (thực)
trong `parameters.txt`, nên các tham số thực được biểu diễn dưới dạng **số
bước nguyên** (`merge_ratio_steps`, `iter_k_steps`, `delta2_steps`,
`delta3_steps`) — `tune.R` tự nhân lại với độ dài bước trước khi truyền cho
exe (xem `STEP_SIZES` trong `tune.R`):

| Tham số | Bước nhảy | Miền số bước | Miền giá trị thực | Số giá trị khả dĩ |
|---|---|---|---|---|
| `merge_ratio` | 0.01 | 5 – 20 | 0.05 – 0.20 | 16 |
| `tabu_cap` | (nguyên sẵn, không đổi) | 5 – 20 | 5 – 20 | 16 |
| `iter_k` | 0.5 | 32 – 44 | 16 – 22 | 13 |
| `delta2` | 0.015 | 1 – 53 | 0.015 – 0.795 | 53 |
| `delta3` | 0.01 | 1 – 80 | 0.01 – 0.80 | 80 |

## 1d. Ràng buộc `delta1 > delta2 > delta3`

Vì `delta1 = 0.8` cố định và miền `delta2` tối đa 0.795 (< 0.8), điều kiện
`delta1 > delta2` luôn tự động đúng. Riêng `delta3 < delta2` **không khai
báo được trực tiếp trong `parameters.txt`** (vì `delta2` và `delta3` dùng 2
bước nhảy khác nhau — 0.015 vs 0.01 — không thể so sánh thẳng "số bước").
Thay vào đó, `target.runner` trong `tune.R` tự kiểm tra sau khi quy đổi ra
giá trị thực: nếu `delta3 >= delta2`, cấu hình bị phạt cost cực lớn
(`1e12`) và **không chạy exe** (tiết kiệm thời gian cho cấu hình chắc chắn
bị loại).

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
- **Instance train**: 12 file trong `tuning/train-instances/` — mỗi size
  (50, 100, 200) lấy 4 instance, mỗi nhóm "grid" (`.10`/`.20`/`.30`/`.40`)
  1 replicate khác nhau cho đa dạng (ví dụ `50.10.1`, `50.20.2`, `50.30.3`,
  `50.40.4`; tương tự cho 100 và 200). Tách riêng khỏi bộ bạn dùng để báo
  cáo kết quả cuối cùng, tránh overfit tham số vào đúng bộ test.
- **Ngân sách**: `MAX_EXPERIMENTS = 22000` lần chạy thuật toán — chỉnh trong
  đầu file `tune.R` (khoá `MAX_EXPERIMENTS`). Nếu muốn test pipeline nhanh
  trước (kiểm tra không lỗi/không crash) thì tạm hạ xuống 500–1000, chạy
  thử, rồi trả lại 22000 cho lần tune "thật".
- **Song song**: `N_PARALLEL = 20` (đặt cho máy 24 nhân, chừa 4 nhân cho hệ
  điều hành/tác vụ khác) — irace tự dùng cluster kiểu PSOCK, chạy được trên
  Windows. Đổi lại nếu chạy trên máy khác có số nhân khác.
- **Timeout mỗi lần chạy**: `PER_RUN_TIMEOUT = 1500` giây (25 phút) — nới
  rộng mạnh vì `iter_k` giờ luôn ở mức cao (16–22, gấp ~2x mặc định gốc
  ~10), làm instance 200 khách hàng chạy rất chậm; cộng thêm hiệu ứng tranh
  chấp CPU khi chạy song song 20 tiến trình.

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
