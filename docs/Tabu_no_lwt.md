# Tabu thuần đối chiếu với Multilevel_Tabu_no_lwt

`src/Tabu_no_lwt.cpp` dùng cấu hình elite 3549:
delta1=0.8, delta2=0.6, delta3=0.1, delta4=0.1;
tabu1=0.07, tabu2=0.11.
Tabu thuần chạy đủ `MAX_ITER` vòng, không dừng sớm theo số vòng không cải thiện.
Cả hai file dùng chung `src/tabu_tenure.h`: mỗi move được chấp nhận lấy
ngẫu nhiên đều một số nguyên trong `[ceil(0.07*n), ceil(0.11*n)]`, ít nhất
là 1, với **n là số khách gốc**, không giảm theo level nén. Move thuận và
move ngược dùng chung tenure của lần chấp nhận đó. Không dùng tabu_factor.
Bộ sinh lấy từ `srand(seed)` hiện có; cùng seed tái lập từng chương trình,
nhưng không buộc hai thuật toán tiêu thụ cùng chuỗi số ngẫu nhiên.

Ngân sách mặc định đối chiếu:

```text
phase_iterations = round(15 * số khách gốc)
total_budget = (2 * 5 + 1) * phase_iterations = 165 * số khách gốc
```

Multilevel có chỉ số level tối đa 5 (level 0..5), tương ứng tối đa 11 lần gọi
Tabu khi nén và tinh chỉnh. Tabu thuần nhận toàn bộ ngân sách này trong một
lần tìm kiếm trên khách gốc. `--max_levels` ở Tabu thuần chỉ xác định ngân
sách đối chiếu, không thực hiện nén. Nếu ML ngừng nén sớm, ngân sách tối đa
này có thể lớn hơn số vòng ML thực chạy.

Biên dịch và chạy cùng seed:

```powershell
g++ -O2 -std=c++17 -o src/Multilevel_Tabu_no_lwt.exe src/Multilevel_Tabu_no_lwt.cpp
g++ -O2 -std=c++17 -o src/Tabu_no_lwt.exe src/Tabu_no_lwt.cpp

.\src\Multilevel_Tabu_no_lwt.exe instances/50.10.1.txt --seed 4321 --iter_k 15 --max_levels 5
.\src\Tabu_no_lwt.exe instances/50.10.1.txt --seed 4321 --iter_k 15 --max_levels 5
```

Hai chương trình vẫn dùng thời gian làm seed khi không truyền `--seed`;
cần truyền seed giống nhau khi so sánh.

Tabu thuần in dòng `CONFIG` với seed, tham số, giới hạn mỗi phase đối chiếu
và tổng ngân sách; dòng `SEARCH` ghi số vòng thực chạy và tenure.
`--budget B` hoặc `--max_iter B` ghi đè **tổng số vòng Tabu thuần**.
Vẫn hỗ trợ cú pháp cũ `Tabu_no_lwt.exe instance.txt B`.

Ví dụ chạy nhanh 100 vòng:

```powershell
.\src\Tabu_no_lwt.exe instances/10.10.1.txt --seed 4321 --budget 100
```
