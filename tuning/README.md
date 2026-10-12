# Tune Multilevel_Tabu_no_lwt bằng irace

## Miền tham số

Alpha 1–4 trong yêu cầu tương ứng `delta1..delta4` trong C++, không phải `alpha1` dùng phạt vi phạm drone trong fitness.

| Tham số | Giá trị | Số lựa chọn |
|---|---|---:|
| max_levels | 4, 5, 6 | 3 |
| merge_ratio | 5–10%, bước 1% (0.05–0.10, bước 0.01) | 6 |
| iter_k | 15, 16, 17, 18 | 4 |
| delta1 | 0.8, cố định | 1 |
| delta2 | 0.4, 0.5, 0.6, 0.7 | 4 |
| delta3 | 0.1, 0.2, 0.3 | 3 |
| tabu1 | 0.05–0.09, bước 0.01 | 5 |
| tabu2 | 0.10–0.12, bước 0.01 | 3 |
| delta4 | 0.1, 0.3, 0.5, 0.7 | 4 |

`merge_ratio_steps` trong `parameters.txt` nhận số nguyên từ 5 đến 10. Runner nhân với 0.01 trước khi truyền `--merge_ratio` cho chương trình C++.

`MAX_ITER = round(iter_k * số khách hàng gốc)`, dùng cùng giới hạn ở các level. Thuật toán có thể dừng sớm theo điều kiện không cải thiện.

`max_levels` là chỉ số level tối đa, bắt đầu từ 0: giá trị 4 cho phép các level 0–4 nếu đủ khả năng nén; số level thực tế có thể ít hơn.

Mỗi move được chọn nhận tenure ngẫu nhiên đều trong khoảng số nguyên:

```text
[max(1, ceil(tabu1 * n)), max(1, ceil(tabu2 * n))]
```

`n` là số khách hàng ở level hiện tại, không tính depot. Move và move đảo nhận cùng tenure. Cả hai cận đều ảnh hưởng việc chọn tenure; làm tròn có thể khiến các hệ số khác nhau cho cùng cận ở level nhỏ. `--seed` cũng điều khiển việc chọn tenure ngẫu nhiên.

## Không gian tìm kiếm và ngân sách

Tổng số tổ hợp: **3 × 6 × 4 × 4 × 3 × 5 × 3 × 4 = 51.840**.

Với 12 instance huấn luyện, vét cạn toàn miền với 1 seed/instance cần **51.840 × 12 = 622.080 lượt chạy**. Irace loại dần cấu hình nên không cần vét cạn.

Cấu hình hiện tại trong `tune.R`: **25.000 lượt chạy**, **20 worker**, timeout **1.500 giây/lượt**. Một lượt tương ứng một cấu hình, một instance và một seed. Ngân sách này không bảo đảm tìm được 5 cấu hình tối ưu.

`minNbSurvival = 5` đặt ngưỡng dừng mỗi race, không bảo đảm trả đúng 5 elite khác nhau. Script lưu toàn bộ elite theo thứ tự irace trả về và xuất tối đa 5 dòng đầu đã đổi sang giá trị thật. Nếu thiếu 5 elite, script cảnh báo.

Ước lượng thời gian lý tưởng:

```text
giờ = ngân_sách * giây_trung_bình / (worker * 3600)
```

Với 25.000 lượt và 20 worker:

| Thời gian trung bình mỗi lượt | Tổng thời gian ước tính |
|---|---:|
| 1 phút | 20,8 giờ |
| 2 phút | 41,7 giờ |
| 5 phút | 104,2 giờ (4,3 ngày) |
| 10 phút | 208,3 giờ (8,7 ngày) |
| 15 phút | 312,5 giờ (13 ngày) |

Các số này chưa tính thời gian điều phối, tải CPU và chờ kết thúc race. Cần đo trên máy chạy thực tế để chọn dòng ước lượng phù hợp. Thu hẹp miền nén không làm ngân sách 25.000 lượt tự giảm theo.

Đánh giá lại 5 cấu hình trên cùng 12 instance × 10 seed cần thêm 600 lượt. Để đánh giá khả năng tổng quát, dùng tập instance riêng chưa dùng để tune.

## Chạy

Từ thư mục gốc, build lại executable sau khi đổi C++:

```powershell
g++ -O2 -std=c++17 -o src/Multilevel_Tabu_no_lwt.exe src/Multilevel_Tabu_no_lwt.cpp
cd tuning
Rscript tune.R
```

Cần R và package irace. Chỉnh `MAX_EXPERIMENTS`, `N_PARALLEL`, `PER_RUN_TIMEOUT` trong `tune.R` khi cần. Runner đọc dòng `IRACE_RESULT`; timeout, crash hoặc thiếu kết quả nhận cost `1e12`.

Kết quả:

- `elite-configurations.csv`: elite với tham số gốc của irace.
- `elite-top5.csv`: tối đa 5 elite đầu với tham số thật.
- `irace-log.Rdata`: log đầy đủ.

Không dùng trung bình chi phí trên các tập instance khác nhau để xếp hạng lại elite.

Tài liệu: [Cấu hình scenario của irace](https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html).
