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
MAX_EXPERIMENTS <- 25000  # Initial search budget, not a guarantee of elite quality.
PER_RUN_TIMEOUT <- 1500   # Seconds per run; inspect timeout rate after a pilot.
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
# Fixed parameters: requested alpha1 maps to delta1, not the fitness penalty alpha1.
# Tenure is sampled within tabu1/tabu2 bounds for each accepted move.
# ---------------------------------------------------------------
FIXED_ARGS <- c("--delta1", "0.8")
STEP_SIZES <- c(merge_ratio = 0.01, tabu1 = 0.01, tabu2 = 0.01)
DIRECT_PARAMS <- c("max_levels", "iter_k", "delta2", "delta3", "delta4")

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

  args <- c(shQuote(experiment$instance), "--seed", as.character(experiment$seed), FIXED_ARGS)
  for (base_name in names(real_values)) {
    args <- c(args, paste0("--", base_name), as.character(real_values[[base_name]]))
  }
  for (name in DIRECT_PARAMS) {
    args <- c(args, paste0("--", name), as.character(conf[[name]]))
  }

  start_time <- Sys.time()
  out <- tryCatch(
    suppressWarnings(system2(EXE_PATH, args = args,
                              stdout = TRUE, stderr = TRUE,
                              timeout = PER_RUN_TIMEOUT)),
    error = function(e) character(0)
  )
  elapsed <- max(as.numeric(difftime(Sys.time(), start_time, units = "secs")), 0.001)  # irace bat buoc time > 0 tuyet doi

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
  minNbSurvival  = 5,
  elitist        = 1,
  logFile        = "irace-log.Rdata",
  seed           = 4321,
  parameters     = parameters
))

cat("\n=== Bat dau tune (maxExperiments =", MAX_EXPERIMENTS, ") ===\n")
elite <- irace(scenario = scenario)

cat("\n=== Cac cau hinh tot nhat (elite configurations) ===\n")
print(elite)

write.csv(elite, file = "elite-configurations.csv", row.names = FALSE)
# Keep irace's returned order; do not rank costs from different instance subsets.
elite_real <- elite
for (name in names(STEP_SIZES)) {
  steps_name <- paste0(name, "_steps")
  elite_real[[name]] <- round(as.numeric(elite_real[[steps_name]]) * STEP_SIZES[[name]], 4)
  elite_real[[steps_name]] <- NULL
}
elite_real$delta1 <- 0.8
write.csv(head(elite_real, 5), file = "elite-top5.csv", row.names = FALSE)
if (nrow(elite) < 5) {
  warning("irace returned fewer than 5 elites; elite-top5.csv contains only actual elites.")
}
cat("\nDa luu vao tuning/elite-configurations.csv\n")
cat("Va log day du (co the mo lai bang irace::plotElite / irace::iraceResults) o tuning/irace-log.Rdata\n")
