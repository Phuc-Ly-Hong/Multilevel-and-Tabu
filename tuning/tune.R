#!/usr/bin/env Rscript
# ============================================================
# tune.R -- Dung irace de tune tham so cho Multilevel_Tabu_no_lwt
# ============================================================
# Cach chay (mo thu muc tuning/ trong terminal roi go):
#   Rscript tune.R
#
# Yeu cau:
#   - R (>= 3.5) da cai, co trong PATH
#   - Goi irace:  install.packages("irace")
#   - Da build:   g++ -O2 -std=c++17 -o src/Multilevel_Tabu_no_lwt.exe src/Multilevel_Tabu_no_lwt.cpp
#
# File nay se:
#   1. Doc parameters.txt (khai bao mien gia tri cac tham so can tune)
#   2. Chay exe lap lai tren cac instance trong train-instances/ voi cac
#      bo tham so khac nhau, doc dong "IRACE_RESULT <so>" ma exe in ra
#   3. Tra ve bo tham so tot nhat (elite configurations)

suppressMessages(library(irace))

# ---------------------------------------------------------------
# Cau hinh nguoi dung co the chinh o day
# ---------------------------------------------------------------
EXE_PATH        <- normalizePath(file.path("..", "src", "Multilevel_Tabu_no_lwt.exe"))
INSTANCES_DIR   <- "train-instances"
MAX_EXPERIMENTS <- 35000  # ngan sach so lan chay thuat toan (6 tham so tune, du du de chac chan lay du top 5 elite)
PER_RUN_TIMEOUT <- 1500   # giay (25 phut), noi rong manh vi iter_k gio luon cao (16-22, gap ~2x mac dinh cu) lam instance 200 rat cham
N_PARALLEL      <- 20     # may 24 nhan, danh 20 nhan chay song song cho irace

if (!file.exists(EXE_PATH)) {
  stop("Khong tim thay executable: ", EXE_PATH,
       "\nHay build truoc bang:\n",
       "  g++ -O2 -std=c++17 -o src/Multilevel_Tabu_no_lwt.exe src/Multilevel_Tabu_no_lwt.cpp")
}

instance_files <- list.files(INSTANCES_DIR, pattern = "\\.txt$", full.names = TRUE)
if (length(instance_files) == 0) stop("Khong co instance nao trong ", INSTANCES_DIR)
cat("So instance dung de tune:", length(instance_files), "\n")

# ---------------------------------------------------------------
# 3 tham so chot cung (khong tune), luon truyen y het cho moi lan chay
# ---------------------------------------------------------------
FIXED_ARGS <- c("--max_levels", "5",
                "--tabu_factor", "0.25",
                "--delta1", "0.8")

# ---------------------------------------------------------------
# Cac tham so con lai duoc irace tune duoi dang "so buoc nguyen" (xem
# parameters.txt) -- can nhan lai voi buoc nhay de ra gia tri thuc, roi
# moi truyen cho exe qua switch tuong ung ("_steps" bi bo di).
# ---------------------------------------------------------------
STEP_SIZES <- c(
  merge_ratio = 0.01,
  iter_k      = 1,
  delta2      = 0.015,
  delta3      = 0.01,
  delta4      = 0.01
)

# ---------------------------------------------------------------
# target-runner: chay 1 lan thuat toan voi 1 bo tham so + 1 instance + 1 seed
# ---------------------------------------------------------------
target.runner <- function(experiment, scenario) {
  conf <- experiment$configuration

  # Tinh gia tri thuc cua tung tham so tu "so buoc"
  real_values <- list()
  for (base_name in names(STEP_SIZES)) {
    steps_name <- paste0(base_name, "_steps")
    if (steps_name %in% names(conf) && !is.na(conf[[steps_name]])) {
      real_values[[base_name]] <- round(STEP_SIZES[[base_name]] * conf[[steps_name]], 4)
    }
  }

  # Rang buoc delta1(=0.8) > delta2 > delta3: delta2 < 0.8 luon dung do
  # mien cho phep (<=0.795), chi can kiem tra delta3 < delta2. Vi pham thi
  # phat that nang, KHONG chay exe (tiet kiem thoi gian).
  if (!is.null(real_values$delta2) && !is.null(real_values$delta3)) {
    if (real_values$delta3 >= real_values$delta2) {
      return(list(cost = 1e12, time = 0))
    }
  }

  args <- c(experiment$instance, "--seed", as.character(experiment$seed), FIXED_ARGS)
  for (base_name in names(real_values)) {
    args <- c(args, paste0("--", base_name), as.character(real_values[[base_name]]))
  }
  if ("tabu_cap" %in% names(conf) && !is.na(conf[["tabu_cap"]])) {
    args <- c(args, "--tabu_cap", as.character(conf[["tabu_cap"]]))
  }

  start_time <- Sys.time()
  out <- tryCatch(
    suppressWarnings(system2(EXE_PATH, args = args,
                              stdout = TRUE, stderr = TRUE,
                              timeout = PER_RUN_TIMEOUT)),
    error = function(e) character(0)
  )
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  result_line <- grep("^IRACE_RESULT ", out, value = TRUE)
  if (length(result_line) == 0) {
    # exe crash / timeout / khong in ket qua -> phat that nang de irace loai cau hinh nay
    return(list(cost = 1e12, time = elapsed))
  }

  cost <- suppressWarnings(as.numeric(sub("^IRACE_RESULT\\s+", "", result_line[1])))
  if (is.na(cost)) cost <- 1e12

  list(cost = cost, time = elapsed)
}

# ---------------------------------------------------------------
# Scenario + chay irace
# ---------------------------------------------------------------
parameters <- readParameters("parameters.txt")

scenario <- defaultScenario(list(
  targetRunner   = target.runner,
  instances      = instance_files,
  maxExperiments = MAX_EXPERIMENTS,
  parallel       = N_PARALLEL,
  logFile        = "irace-log.Rdata",
  seed           = 4321,
  parameters     = parameters
))

cat("\n=== Bat dau tune (maxExperiments =", MAX_EXPERIMENTS, ") ===\n")
elite <- irace(scenario = scenario)

cat("\n=== Cac cau hinh tot nhat (elite configurations) ===\n")
print(elite)

write.csv(elite, file = "elite-configurations.csv", row.names = FALSE)
cat("\nDa luu vao tuning/elite-configurations.csv\n")
cat("Va log day du (co the mo lai bang irace::plotElite / irace::iraceResults) o tuning/irace-log.Rdata\n")
