update_summary_csv <- function(file_path, M) {
  df <- read.csv(file_path)
  df$ECR_SE_Mean <- df$ECR_SD / sqrt(M)
  df$t_stat <- (df$ECR_Mean - 0.90) / df$ECR_SE_Mean
  df$p_coverage <- 2 * (1 - pt(abs(df$t_stat), df = M - 1))
  
  # Reorder columns
  cols <- c("Method", "ECR_Mean", "ECR_SD", "ECR_SE_Mean", "MPIW_Mean", "MPIW_SD", 
            "Winkler_Mean", "Winkler_SD", "t_stat", "p_coverage", "Wilcox_p_vs_Param")
  df <- df[, cols]
  write.csv(df, file_path, row.names = FALSE)
  cat("Updated", file_path, "\n")
  print(df[, c("Method", "ECR_Mean", "ECR_SD", "ECR_SE_Mean", "t_stat", "p_coverage")])
}

update_summary_csv("summary_regime1_gaussian.csv", 100)
update_summary_csv("summary_regime2_student.csv", 100)
update_summary_csv("summary_regime3_hetero.csv", 100)
update_summary_csv("summary_boston_benchmark.csv", 50)
