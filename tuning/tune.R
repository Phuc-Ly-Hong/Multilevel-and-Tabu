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
MAX_EXPERIMENTS <- 25000  # ngan sach so lan chay thuat toan cho lan tune "that"
PER_RUN_TIMEOUT <- 600    # giay, chan neu 1 lan chay bi treo/qua lau (noi rong vi co instance 200 + chay song song 20 tien trinh de cham hon do tranh chap CPU)
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
# Anh xa ten tham so (trong parameters.txt) -> switch dong lenh
# Phai khop voi phan parse "--..." trong main() cua file .cpp
# ---------------------------------------------------------------
switch_map <- c(
  max_levels             = "--max_levels",
  merge_ratio             = "--merge_ratio",
  tabu_factor             = "--tabu_factor",
  tabu_cap                = "--tabu_cap",
  iter_k                  = "--iter_k",
  delta1                  = "--delta1",
  delta2                  = "--delta2",
  delta3                  = "--delta3",
  delta4                  = "--delta4"
)

# ---------------------------------------------------------------
# target-runner: chay 1 lan thuat toan voi 1 bo tham so + 1 instance + 1 seed
# ---------------------------------------------------------------
target.runner <- function(experiment, scenario) {
  conf <- experiment$configuration
  args <- c(experiment$instance, "--seed", as.character(experiment$seed))

  for (pname in names(switch_map)) {
    if (pname %in% names(conf) && !is.na(conf[[pname]])) {
      args <- c(args, switch_map[[pname]], as.character(conf[[pname]]))
    }
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
scenario <- defaultScenario(list(
  targetRunner   = target.runner,
  instances      = instance_files,
  maxExperiments = MAX_EXPERIMENTS,
  parallel       = N_PARALLEL,
  logFile        = "irace-log.Rdata",
  seed           = 4321
))

parameters <- readParameters("parameters.txt")

cat("\n=== Bat dau tune (maxExperiments =", MAX_EXPERIMENTS, ") ===\n")
elite <- irace(scenario = scenario, parameters = parameters)

cat("\n=== Cac cau hinh tot nhat (elite configurations) ===\n")
print(elite)

write.csv(elite, file = "elite-configurations.csv", row.names = FALSE)
cat("\nDa luu vao tuning/elite-configurations.csv\n")
cat("Va log day du (co the mo lai bang irace::plotElite / irace::iraceResults) o tuning/irace-log.Rdata\n")
