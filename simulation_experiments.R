# ==============================================================================
# Distribution-Free Uncertainty Quantification in Machine Learning:
# A Theoretical and Empirical Comparison of Conformal Prediction,
# Bayesian Credible Intervals, and Parametric Bootstrap in R
#
# Simulation Study and Empirical Benchmark Script
# Conforms to JCGS and The R Journal Standards
# Seed: 2026 for Full Scientific Reproducibility
# ==============================================================================

# Ensure local user library is prioritized
.libPaths(c('C:/Users/Yuvraj/AppData/Local/R/win-library/4.6', .libPaths()))

suppressPackageStartupMessages({
  library(ranger)
  library(ggplot2)
  library(MASS)
  library(splines)
})

set.seed(2026)

cat("==============================================================================\n")
cat("Starting Simulation Study: Distribution-Free UQ in Machine Learning\n")
cat("Timestamp: ", as.character(Sys.time()), "\n")
cat("==============================================================================\n\n")

# ------------------------------------------------------------------------------
# 1. EVALUATION METRIC FUNCTIONS
# ------------------------------------------------------------------------------

compute_ecr <- function(y, lower, upper) {
  mean(y >= lower & y <= upper)
}

compute_mpiw <- function(lower, upper) {
  mean(upper - lower)
}

compute_winkler <- function(y, lower, upper, alpha = 0.10) {
  width <- upper - lower
  under <- pmax(0, lower - y)
  over  <- pmax(0, y - upper)
  score <- width + (2 / alpha) * under + (2 / alpha) * over
  mean(score)
}

compute_z_test <- function(ecr, n_test, nominal = 0.90) {
  se <- sqrt(nominal * (1 - nominal) / n_test)
  z <- (ecr - nominal) / se
  p_val <- 2 * (1 - pnorm(abs(z)))
  list(z = z, p_value = p_val)
}

# ------------------------------------------------------------------------------
# 2. DATA GENERATING PROCESS (DGP)
# Y = 2*sin(pi*X1) + X2^2 - 1.5*X3 + epsilon
# ------------------------------------------------------------------------------

generate_synthetic_data <- function(n, regime = c("gaussian", "student_t", "heteroscedastic")) {
  regime <- match.arg(regime)
  X <- matrix(runif(n * 3, -1, 1), nrow = n, ncol = 3)
  colnames(X) <- c("x1", "x2", "x3")
  
  # Ground truth non-linear mean function
  mu <- 2 * sin(pi * X[, 1]) + X[, 2]^2 - 1.5 * X[, 3]
  
  if (regime == "gaussian") {
    eps <- rnorm(n, mean = 0, sd = 1.0)
  } else if (regime == "student_t") {
    # Heavy-tailed Student-t with df = 3 (infinite kurtosis)
    eps <- rt(n, df = 3)
  } else if (regime == "heteroscedastic") {
    # Heteroscedastic and right-skewed noise:
    # Exp(1) - 1 has mean 0, variance 1, skewness 2
    raw_exp <- rexp(n, rate = 1) - 1
    sigma_x <- 1 + 2 * abs(X[, 1])
    eps <- sigma_x * raw_exp
  }
  
  y <- mu + eps
  data.frame(y = y, x1 = X[, 1], x2 = X[, 2], x3 = X[, 3])
}

# ------------------------------------------------------------------------------
# 3. PREDICTION INTERVAL METHODS
# ------------------------------------------------------------------------------

# Method 1: Parametric Polynomial/Spline OLS Interval
fit_parametric_ols <- function(train_df, test_df, alpha = 0.10) {
  # Fits a flexible polynomial/spline model
  fit <- lm(y ~ poly(x1, 3) + poly(x2, 2) + x3, data = train_df)
  preds <- predict(fit, newdata = test_df, interval = "prediction", level = 1 - alpha)
  list(lower = preds[, "lwr"], upper = preds[, "upr"])
}

# Method 2: Bayesian Conjugate Normal-Inverse-Gamma Regression
fit_bayesian_nig <- function(train_df, test_df, alpha = 0.10) {
  # Fits polynomial basis model under conjugate Normal-Inverse-Gamma prior
  # Under diffuse/Jeffreys prior, the exact marginal posterior predictive distribution 
  # for a new observation y* is Student-t with nu = n - p degrees of freedom.
  fit <- lm(y ~ poly(x1, 3) + poly(x2, 2) + x3, data = train_df)
  pred_obj <- predict(fit, newdata = test_df, se.fit = TRUE)
  
  nu <- fit$df.residual
  s2 <- sum(residuals(fit)^2) / nu
  
  # Leverage for test points: h_ii = (se.fit / residual.scale)^2
  h_te <- (pred_obj$se.fit / pred_obj$residual.scale)^2
  pred_se <- sqrt(s2 * (1 + h_te))
  
  # Exact analytical equal-tailed Bayesian Credible Interval
  t_crit <- qt(1 - alpha / 2, df = nu)
  lower <- pred_obj$fit - t_crit * pred_se
  upper <- pred_obj$fit + t_crit * pred_se
  
  list(lower = as.vector(lower), upper = as.vector(upper))
}

# Method 3: Parametric Bootstrap Prediction Interval
fit_parametric_bootstrap <- function(train_df, test_df, formula = y ~ poly(x1, 3) + poly(x2, 2) + x3, alpha = 0.10, B = 250) {
  fit_base <- lm(formula, data = train_df)
  terms_base <- terms(fit_base)
  
  X_tr <- model.matrix(terms_base, data = train_df)
  y_tr <- train_df$y
  n <- nrow(X_tr)
  p <- ncol(X_tr)
  
  XtX_inv <- solve(crossprod(X_tr))
  beta_hat <- XtX_inv %*% crossprod(X_tr, y_tr)
  sigma_hat <- sqrt(sum((y_tr - X_tr %*% beta_hat)^2) / (n - p))
  
  mf_te <- model.frame(delete.response(terms_base), data = test_df)
  X_te <- model.matrix(delete.response(terms_base), mf_te)
  n_te <- nrow(X_te)
  
  # Vectorized parametric bootstrap draws under Gaussian assumption
  eps_boot_tr <- matrix(rnorm(n * B, mean = 0, sd = sigma_hat), nrow = n, ncol = B)
  Y_boot <- as.vector(X_tr %*% beta_hat) + eps_boot_tr
  
  Beta_boot <- XtX_inv %*% crossprod(X_tr, Y_boot)
  
  eps_boot_te <- matrix(rnorm(n_te * B, mean = 0, sd = sigma_hat), nrow = n_te, ncol = B)
  Y_te_pred <- X_te %*% Beta_boot + eps_boot_te
  
  lower <- apply(Y_te_pred, 1, quantile, probs = alpha / 2, names = FALSE)
  upper <- apply(Y_te_pred, 1, quantile, probs = 1 - alpha / 2, names = FALSE)
  
  list(lower = as.vector(lower), upper = as.vector(upper))
}

# Method 4: Split Conformal Random Forest
fit_split_conformal <- function(train_df, cal_df, test_df, alpha = 0.10) {
  n2 <- nrow(cal_df)
  rf <- ranger(y ~ ., data = train_df, num.trees = 100, verbose = FALSE)
  
  # Calibration non-conformity scores: absolute residuals
  y_hat_cal <- predict(rf, data = cal_df)$predictions
  cal_scores <- abs(cal_df$y - y_hat_cal)
  
  # Exact conformal quantile index: ceiling((n2 + 1) * (1 - alpha))
  k <- ceiling((n2 + 1) * (1 - alpha))
  k <- min(k, n2)
  q_hat <- sort(cal_scores)[k]
  
  # Test predictions
  y_hat_test <- predict(rf, data = test_df)$predictions
  lower <- y_hat_test - q_hat
  upper <- y_hat_test + q_hat
  
  list(lower = lower, upper = upper, q_hat = q_hat)
}

# Method 5: Conformalized Quantile Regression (CQR) via Quantile Random Forest
fit_cqr <- function(train_df, cal_df, test_df, alpha = 0.10) {
  n2 <- nrow(cal_df)
  # Fit Quantile Random Forest on proper training set
  qrf <- ranger(y ~ ., data = train_df, num.trees = 100, quantreg = TRUE, verbose = FALSE)
  
  # Predict lower and upper quantiles on calibration set
  q_levels <- c(alpha / 2, 1 - alpha / 2)
  cal_preds <- predict(qrf, data = cal_df, type = "quantiles", quantiles = q_levels)$predictions
  q_low_cal  <- cal_preds[, 1]
  q_high_cal <- cal_preds[, 2]
  
  # Signed conformity scores: E_i = max(q_low(X_i) - Y_i, Y_i - q_high(X_i))
  E_scores <- pmax(q_low_cal - cal_df$y, cal_df$y - q_high_cal)
  
  # Conformal quantile adjustment
  k <- ceiling((n2 + 1) * (1 - alpha))
  k <- min(k, n2)
  Q_conf <- sort(E_scores)[k]
  
  # Test predictions
  test_preds <- predict(qrf, data = test_df, type = "quantiles", quantiles = q_levels)$predictions
  lower <- test_preds[, 1] - Q_conf
  upper <- test_preds[, 2] + Q_conf
  
  list(lower = lower, upper = upper, Q_conf = Q_conf)
}

# ------------------------------------------------------------------------------
# 4. MONTE CARLO EXPERIMENTAL RUNNER (EXPERIMENT A)
# ------------------------------------------------------------------------------

run_monte_carlo <- function(M = 100, N = 1500, n1 = 750, n2 = 375, alpha = 0.10) {
  regimes <- c("gaussian", "student_t", "heteroscedastic")
  methods <- c("Parametric_OLS", "Parametric_Bootstrap", "Bayesian_NIG", "Split_Conformal", "CQR")
  
  results_list <- list()
  counter <- 1
  
  for (reg in regimes) {
    cat(sprintf("\n>>> Executing Monte Carlo Replications for Regime: %s (M = %d) ...\n", reg, M))
    
    # Preallocate metrics matrix for M runs: [ECR, MPIW, Winkler]
    metrics_m <- array(NA, dim = c(M, length(methods), 3),
                       dimnames = list(1:M, methods, c("ECR", "MPIW", "Winkler")))
    
    for (m in 1:M) {
      if (m %% 25 == 0) cat(sprintf("    Replication %d / %d ...\n", m, M))
      
      data_full <- generate_synthetic_data(N, regime = reg)
      
      # Three-way partition: Proper Train (n1), Calibration (n2), Test (n_test)
      train_df <- data_full[1:n1, ]
      cal_df   <- data_full[(n1 + 1):(n1 + n2), ]
      test_df  <- data_full[(n1 + n2 + 1):N, ]
      
      # Combined train + cal for parametric/Bayesian/bootstrap to ensure fair sample size comparison
      train_cal_df <- rbind(train_df, cal_df)
      
      # 1. Parametric OLS
      res_param <- fit_parametric_ols(train_cal_df, test_df, alpha = alpha)
      metrics_m[m, "Parametric_OLS", "ECR"]     <- compute_ecr(test_df$y, res_param$lower, res_param$upper)
      metrics_m[m, "Parametric_OLS", "MPIW"]    <- compute_mpiw(res_param$lower, res_param$upper)
      metrics_m[m, "Parametric_OLS", "Winkler"] <- compute_winkler(test_df$y, res_param$lower, res_param$upper, alpha)
      
      # 2. Parametric Bootstrap
      res_boot <- fit_parametric_bootstrap(train_cal_df, test_df, alpha = alpha, B = 250)
      metrics_m[m, "Parametric_Bootstrap", "ECR"]     <- compute_ecr(test_df$y, res_boot$lower, res_boot$upper)
      metrics_m[m, "Parametric_Bootstrap", "MPIW"]    <- compute_mpiw(res_boot$lower, res_boot$upper)
      metrics_m[m, "Parametric_Bootstrap", "Winkler"] <- compute_winkler(test_df$y, res_boot$lower, res_boot$upper, alpha)
      
      # 3. Bayesian NIG
      res_bayes <- fit_bayesian_nig(train_cal_df, test_df, alpha = alpha)
      metrics_m[m, "Bayesian_NIG", "ECR"]     <- compute_ecr(test_df$y, res_bayes$lower, res_bayes$upper)
      metrics_m[m, "Bayesian_NIG", "MPIW"]    <- compute_mpiw(res_bayes$lower, res_bayes$upper)
      metrics_m[m, "Bayesian_NIG", "Winkler"] <- compute_winkler(test_df$y, res_bayes$lower, res_bayes$upper, alpha)
      
      # 4. Split Conformal
      res_split <- fit_split_conformal(train_df, cal_df, test_df, alpha = alpha)
      metrics_m[m, "Split_Conformal", "ECR"]     <- compute_ecr(test_df$y, res_split$lower, res_split$upper)
      metrics_m[m, "Split_Conformal", "MPIW"]    <- compute_mpiw(res_split$lower, res_split$upper)
      metrics_m[m, "Split_Conformal", "Winkler"] <- compute_winkler(test_df$y, res_split$lower, res_split$upper, alpha)
      
      # 5. CQR
      res_cqr <- fit_cqr(train_df, cal_df, test_df, alpha = alpha)
      metrics_m[m, "CQR", "ECR"]     <- compute_ecr(test_df$y, res_cqr$lower, res_cqr$upper)
      metrics_m[m, "CQR", "MPIW"]    <- compute_mpiw(res_cqr$lower, res_cqr$upper)
      metrics_m[m, "CQR", "Winkler"] <- compute_winkler(test_df$y, res_cqr$lower, res_cqr$upper, alpha)
    }
    
    results_list[[reg]] <- metrics_m
  }
  
  results_list
}

mc_results <- run_monte_carlo(M = 100, N = 1500, n1 = 750, n2 = 375, alpha = 0.10)

# ------------------------------------------------------------------------------
# 5. REAL-WORLD BENCHMARK STUDY (EXPERIMENT B: Boston Housing)
# ------------------------------------------------------------------------------

run_boston_benchmark <- function(B = 50, alpha = 0.10) {
  cat("\n>>> Executing Real-World Benchmark on MASS::Boston (B = 50 splits) ...\n")
  data(Boston, package = "MASS")
  df <- Boston
  names(df)[names(df) == "medv"] <- "y"
  N <- nrow(df)
  
  methods <- c("Parametric_OLS", "Parametric_Bootstrap", "Bayesian_NIG", "Split_Conformal", "CQR")
  metrics_b <- array(NA, dim = c(B, length(methods), 3),
                     dimnames = list(1:B, methods, c("ECR", "MPIW", "Winkler")))
  
  for (b in 1:B) {
    perm <- sample(N)
    n1 <- floor(0.50 * N)
    n2 <- floor(0.25 * N)
    
    idx_train <- perm[1:n1]
    idx_cal   <- perm[(n1 + 1):(n1 + n2)]
    idx_test  <- perm[(n1 + n2 + 1):N]
    
    train_df <- df[idx_train, ]
    cal_df   <- df[idx_cal, ]
    test_df  <- df[idx_test, ]
    train_cal_df <- rbind(train_df, cal_df)
    
    # 1. Parametric linear model
    fit_lm <- lm(y ~ ., data = train_cal_df)
    pr_lm <- predict(fit_lm, newdata = test_df, interval = "prediction", level = 1 - alpha)
    metrics_b[b, "Parametric_OLS", "ECR"]     <- compute_ecr(test_df$y, pr_lm[, "lwr"], pr_lm[, "upr"])
    metrics_b[b, "Parametric_OLS", "MPIW"]    <- compute_mpiw(pr_lm[, "lwr"], pr_lm[, "upr"])
    metrics_b[b, "Parametric_OLS", "Winkler"] <- compute_winkler(test_df$y, pr_lm[, "lwr"], pr_lm[, "upr"], alpha)
    
    # 2. Parametric Bootstrap
    res_boot <- fit_parametric_bootstrap(train_cal_df, test_df, formula = y ~ ., alpha = alpha, B = 250)
    metrics_b[b, "Parametric_Bootstrap", "ECR"]     <- compute_ecr(test_df$y, res_boot$lower, res_boot$upper)
    metrics_b[b, "Parametric_Bootstrap", "MPIW"]    <- compute_mpiw(res_boot$lower, res_boot$upper)
    metrics_b[b, "Parametric_Bootstrap", "Winkler"] <- compute_winkler(test_df$y, res_boot$lower, res_boot$upper, alpha)
    
    # 3. Bayesian NIG
    X_tr <- model.matrix(y ~ ., train_cal_df)
    y_tr <- train_cal_df$y
    X_te <- model.matrix(y ~ ., test_df)
    XtX_inv <- solve(crossprod(X_tr))
    beta_hat <- XtX_inv %*% crossprod(X_tr, y_tr)
    nu <- nrow(X_tr) - ncol(X_tr)
    s2 <- sum((y_tr - X_tr %*% beta_hat)^2) / nu
    lev_te <- rowSums((X_te %*% XtX_inv) * X_te)
    se_te <- sqrt(s2 * (1 + lev_te))
    t_c <- qt(1 - alpha / 2, df = nu)
    mu_te <- as.vector(X_te %*% beta_hat)
    b_lwr <- mu_te - t_c * se_te
    b_upr <- mu_te + t_c * se_te
    metrics_b[b, "Bayesian_NIG", "ECR"]     <- compute_ecr(test_df$y, b_lwr, b_upr)
    metrics_b[b, "Bayesian_NIG", "MPIW"]    <- compute_mpiw(b_lwr, b_upr)
    metrics_b[b, "Bayesian_NIG", "Winkler"] <- compute_winkler(test_df$y, b_lwr, b_upr, alpha)
    
    # 4. Split Conformal
    res_sp <- fit_split_conformal(train_df, cal_df, test_df, alpha)
    metrics_b[b, "Split_Conformal", "ECR"]     <- compute_ecr(test_df$y, res_sp$lower, res_sp$upper)
    metrics_b[b, "Split_Conformal", "MPIW"]    <- compute_mpiw(res_sp$lower, res_sp$upper)
    metrics_b[b, "Split_Conformal", "Winkler"] <- compute_winkler(test_df$y, res_sp$lower, res_sp$upper, alpha)
    
    # 5. CQR
    res_cq <- fit_cqr(train_df, cal_df, test_df, alpha)
    metrics_b[b, "CQR", "ECR"]     <- compute_ecr(test_df$y, res_cq$lower, res_cq$upper)
    metrics_b[b, "CQR", "MPIW"]    <- compute_mpiw(res_cq$lower, res_cq$upper)
    metrics_b[b, "CQR", "Winkler"] <- compute_winkler(test_df$y, res_cq$lower, res_cq$upper, alpha)
  }
  
  metrics_b
}

boston_results <- run_boston_benchmark(B = 50, alpha = 0.10)

# ------------------------------------------------------------------------------
# 6. STATISTICAL SUMMARIES & HYPOTHESIS TESTING
# ------------------------------------------------------------------------------

summarize_regime <- function(arr, regime_name, nominal = 0.90) {
  cat(sprintf("\n==============================================================================\n"))
  cat(sprintf("SUMMARY TABLE: %s (Nominal Coverage = %.2f)\n", toupper(regime_name), nominal))
  cat(sprintf("==============================================================================\n"))
  methods <- dimnames(arr)[[2]]
  M_reps <- dim(arr)[1]
  
  summary_df <- data.frame(
    Method = character(),
    ECR_Mean = numeric(),
    ECR_SD = numeric(),
    ECR_SE_Mean = numeric(),
    MPIW_Mean = numeric(),
    MPIW_SD = numeric(),
    Winkler_Mean = numeric(),
    Winkler_SD = numeric(),
    t_stat = numeric(),
    p_coverage = numeric(),
    Wilcox_p_vs_Param = numeric(),
    stringsAsFactors = FALSE
  )
  
  param_winkler <- arr[, "Parametric_OLS", "Winkler"]
  
  for (m in methods) {
    ecr_vals <- arr[, m, "ECR"]
    mpiw_vals <- arr[, m, "MPIW"]
    wink_vals <- arr[, m, "Winkler"]
    
    mean_ecr <- mean(ecr_vals)
    sd_ecr   <- sd(ecr_vals)
    se_ecr_mean <- sd_ecr / sqrt(M_reps)
    
    mean_mpiw <- mean(mpiw_vals)
    sd_mpiw   <- sd(mpiw_vals)
    mean_wink <- mean(wink_vals)
    sd_wink   <- sd(wink_vals)
    
    # Rigorous two-sided Student's t-test on Monte Carlo mean coverage across replications
    t_val <- (mean_ecr - nominal) / se_ecr_mean
    p_val <- 2 * (1 - pt(abs(t_val), df = M_reps - 1))
    
    # Paired Wilcoxon test on Winkler score vs. Parametric OLS
    if (m == "Parametric_OLS") {
      wilcox_p <- NA
    } else {
      wilcox_p <- wilcox.test(wink_vals, param_winkler, paired = TRUE)$p.value
    }
    
    cat(sprintf("%-16s | ECR: %.4f (%.4f) | SE: %.5f | MPIW: %6.3f (%5.3f) | Wink: %6.3f (%5.3f) | t: %+6.2f (p=%.4e) | Wilc_p: %.2e\n",
                m, mean_ecr, sd_ecr, se_ecr_mean, mean_mpiw, sd_mpiw, mean_wink, sd_wink, t_val, p_val, ifelse(is.na(wilcox_p), 1.0, wilcox_p)))
    
    summary_df <- rbind(summary_df, data.frame(
      Method = m,
      ECR_Mean = mean_ecr,
      ECR_SD = sd_ecr,
      ECR_SE_Mean = se_ecr_mean,
      MPIW_Mean = mean_mpiw,
      MPIW_SD = sd_mpiw,
      Winkler_Mean = mean_wink,
      Winkler_SD = sd_wink,
      t_stat = t_val,
      p_coverage = p_val,
      Wilcox_p_vs_Param = wilcox_p,
      stringsAsFactors = FALSE
    ))
  }
  
  summary_df
}

summary_gaussian <- summarize_regime(mc_results$gaussian, "Regime I: Homoscedastic Gaussian")
summary_student  <- summarize_regime(mc_results$student_t, "Regime II: Heavy-Tailed Student-t(3)")
summary_hetero   <- summarize_regime(mc_results$heteroscedastic, "Regime III: Heteroscedastic & Skewed")
summary_boston   <- summarize_regime(boston_results, "Experiment B: Boston Housing Benchmark")

# Save summary tables to CSV
write.csv(summary_gaussian, "summary_regime1_gaussian.csv", row.names = FALSE)
write.csv(summary_student,  "summary_regime2_student.csv",  row.names = FALSE)
write.csv(summary_hetero,   "summary_regime3_hetero.csv",   row.names = FALSE)
write.csv(summary_boston,   "summary_boston_benchmark.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 7. PUBLICATION-QUALITY GGPLOT2 VISUALIZATIONS
# ------------------------------------------------------------------------------

cat("\n>>> Generating Publication-Grade Figures ...\n")

# Figure 1: Prediction Bands across X1 under Heteroscedastic Skewed Noise
set.seed(42)
n_vis <- 300
vis_data <- generate_synthetic_data(n_vis, regime = "heteroscedastic")
vis_data <- vis_data[order(vis_data$x1), ]

train_vis <- generate_synthetic_data(750, regime = "heteroscedastic")
cal_vis   <- generate_synthetic_data(375, regime = "heteroscedastic")
train_cal_vis <- rbind(train_vis, cal_vis)

# Fit models
res_param_vis <- fit_parametric_ols(train_cal_vis, vis_data, alpha = 0.10)
res_split_vis <- fit_split_conformal(train_vis, cal_vis, vis_data, alpha = 0.10)
res_cqr_vis   <- fit_cqr(train_vis, cal_vis, vis_data, alpha = 0.10)

df_bands <- data.frame(
  x1 = rep(vis_data$x1, 3),
  y  = rep(vis_data$y, 3),
  lower = c(res_param_vis$lower, res_split_vis$lower, res_cqr_vis$lower),
  upper = c(res_param_vis$upper, res_split_vis$upper, res_cqr_vis$upper),
  Method = factor(rep(c("Parametric (Polynomial OLS)", "Split Conformal (Random Forest)", "CQR (Quantile Random Forest)"),
                      each = n_vis),
                  levels = c("Parametric (Polynomial OLS)", "Split Conformal (Random Forest)", "CQR (Quantile Random Forest)"))
)

p1 <- ggplot(df_bands, aes(x = x1)) +
  geom_point(aes(y = y), color = "gray30", alpha = 0.45, size = 1.3) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = Method), alpha = 0.35) +
  geom_line(aes(y = (lower + upper)/2, color = Method), linewidth = 0.8) +
  facet_wrap(~ Method, ncol = 3) +
  scale_fill_manual(values = c("#D95F02", "#7570B3", "#1B9E77")) +
  scale_color_manual(values = c("#D95F02", "#7570B3", "#1B9E77")) +
  labs(
    title = "Figure 1: Empirical Prediction Bands under Heteroscedastic Skewed Noise",
    subtitle = expression(paste("True DGP: ", Y == 2*sin(pi*X[1]) + X[2]^2 - 1.5*X[3] + (1 + 2*abs(X[1]))*(Exp(1) - 1), " (Nominal Coverage = 90%)")),
    x = expression(Covariate ~ X[1]),
    y = "Response Y"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, color = "gray25"),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank()
  )

ggsave("fig1_prediction_bands.png", plot = p1, width = 11, height = 4.2, dpi = 300)
cat("Saved: fig1_prediction_bands.png\n")

# Figure 2: Coverage Distribution across M=100 Monte Carlo Replications
cov_df_list <- list()
for (reg in names(mc_results)) {
  for (m in c("Parametric_OLS", "Parametric_Bootstrap", "Bayesian_NIG", "Split_Conformal", "CQR")) {
    reg_label <- switch(reg,
                        "gaussian" = "Regime I: Gaussian",
                        "student_t" = "Regime II: Student-t(3)",
                        "heteroscedastic" = "Regime III: Heteroscedastic")
    cov_df_list[[length(cov_df_list) + 1]] <- data.frame(
      Regime = reg_label,
      Method = m,
      Coverage = mc_results[[reg]][, m, "ECR"]
    )
  }
}
cov_df_all <- do.call(rbind, cov_df_list)
cov_df_all$Method <- factor(cov_df_all$Method,
                           levels = c("Parametric_OLS", "Parametric_Bootstrap", "Bayesian_NIG", "Split_Conformal", "CQR"),
                           labels = c("Parametric OLS", "Parametric Bootstrap", "Bayesian NIG", "Split Conformal", "CQR"))

p2 <- ggplot(cov_df_all, aes(x = Method, y = Coverage, fill = Method)) +
  geom_hline(yintercept = 0.90, linetype = "dashed", color = "firebrick3", linewidth = 0.9) +
  geom_boxplot(alpha = 0.75, outlier.size = 1.0, width = 0.6) +
  facet_wrap(~ Regime, ncol = 3) +
  scale_fill_manual(values = c("#D95F02", "#FC8D62", "#8DA0CB", "#7570B3", "#1B9E77")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0.70, 0.98)) +
  labs(
    title = "Figure 2: Empirical Coverage Distributions across M = 100 Monte Carlo Replications",
    subtitle = "Dashed red line indicates nominal 90% coverage level. Boxplots show median, IQR, and outliers.",
    x = "Quantification Paradigm",
    y = "Empirical Test Coverage Rate (ECR)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 25, hjust = 1, face = "bold"),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, color = "gray25"),
    strip.text = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank()
  )

ggsave("fig2_coverage_distribution.png", plot = p2, width = 11, height = 4.8, dpi = 300)
cat("Saved: fig2_coverage_distribution.png\n")

# Figure 3: Winkler Score vs. Calibration Sample Size n2 Learning Curve
cat("Computing calibration sample size learning curve for Figure 3 ...\n")
n2_grid <- c(25, 50, 100, 200, 350, 500, 750)
curve_reps <- 30
curve_df_list <- list()

for (n2_val in n2_grid) {
  for (r in 1:curve_reps) {
    dat <- generate_synthetic_data(750 + n2_val + 375, regime = "heteroscedastic")
    tr_df  <- dat[1:750, ]
    cl_df  <- dat[751:(750 + n2_val), ]
    te_df  <- dat[(750 + n2_val + 1):nrow(dat), ]
    
    # Split conformal
    sp_res <- fit_split_conformal(tr_df, cl_df, te_df, alpha = 0.10)
    w_sp <- compute_winkler(te_df$y, sp_res$lower, sp_res$upper, 0.10)
    
    # CQR
    cq_res <- fit_cqr(tr_df, cl_df, te_df, alpha = 0.10)
    w_cq <- compute_winkler(te_df$y, cq_res$lower, cq_res$upper, 0.10)
    
    curve_df_list[[length(curve_df_list) + 1]] <- data.frame(
      n2 = n2_val,
      Method = "Split Conformal",
      Winkler = w_sp
    )
    curve_df_list[[length(curve_df_list) + 1]] <- data.frame(
      n2 = n2_val,
      Method = "CQR",
      Winkler = w_cq
    )
  }
}

curve_df <- do.call(rbind, curve_df_list)
curve_summary <- aggregate(Winkler ~ n2 + Method, data = curve_df, FUN = function(x) c(mean = mean(x), se = sd(x)/sqrt(length(x))))
curve_summary <- data.frame(
  n2 = curve_summary$n2,
  Method = curve_summary$Method,
  Mean = curve_summary$Winkler[, "mean"],
  SE = curve_summary$Winkler[, "se"]
)

p3 <- ggplot(curve_summary, aes(x = n2, y = Mean, color = Method, fill = Method)) +
  geom_ribbon(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), alpha = 0.25, color = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.5) +
  scale_color_manual(values = c("CQR" = "#1B9E77", "Split Conformal" = "#7570B3")) +
  scale_fill_manual(values = c("CQR" = "#1B9E77", "Split Conformal" = "#7570B3")) +
  scale_x_continuous(trans = "log10", breaks = n2_grid) +
  labs(
    title = "Figure 3: Interval Efficiency (Winkler Score) vs. Calibration Sample Size n2",
    subtitle = "Evaluation under Regime III (Heteroscedastic). Shaded ribbon shows 95% confidence intervals over 30 runs.",
    x = expression(Calibration ~ Sample ~ Size ~ n[2] ~ (log[10] ~ scale)),
    y = "Mean Winkler Score (Lower is More Efficient)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = c(0.82, 0.82),
    legend.background = element_rect(fill = alpha("white", 0.8), color = "gray80"),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9.5, color = "gray25"),
    panel.grid.minor = element_blank()
  )

ggsave("fig3_winkler_calibration_curve.png", plot = p3, width = 8, height = 4.8, dpi = 300)
cat("Saved: fig3_winkler_calibration_curve.png\n")

cat("\n==============================================================================\n")
cat("SIMULATION & BENCHMARK COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")
