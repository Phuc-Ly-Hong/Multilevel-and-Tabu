#!/usr/bin/env Rscript
# ============================================================
# top-configs.R -- Lay top N cau hinh tot nhat tu toan bo log irace,
# khong chi rieng nhom "elite" cuoi cung (~3-5 cai)
# ============================================================
# Cach chay (dung trong thu muc tuning/):
#   Rscript top-configs.R
#
# Doc:
#   - irace-log.Rdata (toan bo lich su chay) -> tinh fitness trung binh
#     cua MOI cau hinh tung duoc thu, tren tat ca instance da test no
#   - elite-configurations.csv (ket qua cuoi cung irace bao cao) -> danh
#     dau cau hinh nao thuc su "song sot" den het qua trinh tune

TOP_N <- 10
MIN_INSTANCES <- 5   # chi tin cau hinh da duoc test tren >= 5 instance

load("irace-log.Rdata")

exp_matrix <- iraceResults$experiments        # hang = instance, cot = ID cau hinh
configs    <- iraceResults$allConfigurations  # tham so cua tung cau hinh theo ID

mean_cost <- colMeans(exp_matrix, na.rm = TRUE)
n_tested  <- colSums(!is.na(exp_matrix))

ranking <- data.frame(
  ID = as.integer(names(mean_cost)),
  mean_cost = as.numeric(mean_cost),
  n_instances_tested = as.integer(n_tested)
)

# Doc danh sach cau hinh da song sot den cuoi (file elite-configurations.csv
# duoc ghi boi tune.R sau khi chay xong)
final_elite_ids <- integer(0)
if (file.exists("elite-configurations.csv")) {
  elite_csv <- read.csv("elite-configurations.csv")
  id_col <- if (".ID." %in% names(elite_csv)) ".ID." else names(elite_csv)[1]
  final_elite_ids <- as.integer(elite_csv[[id_col]])
}
ranking$song_sot_den_cuoi <- ranking$ID %in% final_elite_ids

# Chi xep hang nhung cau hinh co du du lieu (test tren >= MIN_INSTANCES)
ranking_fair <- subset(ranking, n_instances_tested >= MIN_INSTANCES)
ranking_fair <- ranking_fair[order(ranking_fair$mean_cost), ]

top_n <- head(ranking_fair, TOP_N)
top_n <- merge(top_n, configs, by.x = "ID", by.y = ".ID.")
top_n <- top_n[order(top_n$mean_cost), ]

# Sap xep lai cot cho de doc
front_cols <- c("ID", "mean_cost", "n_instances_tested", "song_sot_den_cuoi")
other_cols <- setdiff(names(top_n), front_cols)
top_n <- top_n[, c(front_cols, other_cols)]

cat("\n=== TOP", TOP_N, "cau hinh tot nhat (tinh tren toan bo log,",
    "chi xet cau hinh test >=", MIN_INSTANCES, "instance) ===\n\n")
print(top_n, row.names = FALSE)

write.csv(top_n, "top-configurations.csv", row.names = FALSE)
cat("\nDa luu vao tuning/top-configurations.csv\n")
cat("\nCot 'song_sot_den_cuoi' = TRUE nghia la cau hinh nay nam trong nhom\n")
cat("elite ma irace giu lai den het qua trinh tune (dang tin cay nhat).\n")
cat("Cot 'n_instances_tested' cang cao thi mean_cost cang dang tin (test\n")
cat("tren nhieu instance hon), cang thap thi con it bang chung, can can trong.\n")
