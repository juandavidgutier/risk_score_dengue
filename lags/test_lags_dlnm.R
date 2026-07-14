#!/usr/bin/env Rscript
"""
test_lags_dlnm.R
=================
Strategy 4: Uses DLNM (Distributed Lag Non-linear Models) to automatically
estimate which lags of potential_intro and R0_rel are important
for predicting dengue cases.

IMPORTANT NOTE:
- The DLNM model is fitted with ALL municipalities in Colombia to
  maximize training data and obtain more robust lag coefficient estimates.
- Final metrics are evaluated ONLY in municipalities with altitude >2000m
  (for direct comparison with S1-S3 strategies).
- Lag importance is reported for both components using all municipalities.

Methodology:
  1) Load data (potential_intro, R0_rel, cases, risk from ALL municipalities)
  2) For each municipality, build crossbasis with lags 0-3
  3) Fit binomial GLM with DLNM for each municipality
  4) Predict risk (outbreak probability) for each municipality-month
  5) Evaluate metrics ONLY in municipalities >2000m (AUC-ROC, AUC-PR, etc.)
  6) Identify the most important lags via crossreduce
  7) Save results in JSON and CSV

Usage:
  Rscript test_lags_dlnm.R
  (Requires R >= 4.0 with packages: dlnm, jsonlite, splines, pROC, PRROC)
"""

# --- 0. CONFIG ---
BASE_DIR   <- "C:/"
OUTPUT_DIR <- file.path(BASE_DIR, "lags")

risk_file  <- file.path(BASE_DIR, "risk_score", "risk_monthly.csv.gz")     # ALL municipalities
cases_file <- file.path(BASE_DIR, "cases", "cases_dengue.csv")
intro_file <- file.path(BASE_DIR, "potential_introduction", "potential_intro_monthly.csv.gz")
r0_file    <- file.path(BASE_DIR, "R0", "R0_rel_monthly.csv")

cat(rep("=", 70), "\n", sep="")
cat("DLNM LAG STRATEGY (Strategy 4)\n")
cat(rep("=", 70), "\n", sep="")

# --- 1. LOAD PACKAGES ---
cat("\n[1/6] Loading packages...\n")
suppressPackageStartupMessages({
  library(dlnm)
  library(jsonlite)
  library(splines)
  library(pROC)
})
has_PRROC <- requireNamespace("PRROC", quietly=TRUE)
if (!has_PRROC) cat("  [WARN] PRROC package not installed. AUC-PR will not be computed.\n")

# --- 2. LOAD DATA ---
cat("[2/6] Loading data...\n")

# Risk (all municipalities, columns: population, altitude_min, risk, etc.)
risk <- read.csv(risk_file, stringsAsFactors=FALSE)
risk$Year_month <- as.character(risk$Year_month)
n_munis_total <- length(unique(risk$DANE))
n_high <- sum(risk$altitude_min[!duplicated(risk$DANE)] > 2000, na.rm=TRUE)
cat(sprintf("  Risk (all munis): %d records, %d municipalities (%d > 2000m)\n",
    nrow(risk), n_munis_total, n_high))

# Cases
cases <- read.csv(cases_file, stringsAsFactors=FALSE)
cases$Year_month <- as.character(cases$Year_month)
cat(sprintf("  Cases: %d records, %d municipalities\n", nrow(cases), length(unique(cases$DANE))))

# Potential_intro
intro <- read.csv(gzfile(intro_file), stringsAsFactors=FALSE)
intro$Year_month <- as.character(intro$Year_month)
cat(sprintf("  Potential intro: %d records\n", nrow(intro)))

# R0_rel
r0 <- read.csv(r0_file, stringsAsFactors=FALSE)
r0$Year_month <- as.character(r0$Year_month)
cat(sprintf("  R0_rel: %d records\n", nrow(r0)))

# --- 3. MERGE AND PREPARE ---
cat("[3/6] Merging data...\n")

df <- merge(risk, cases[, c("DANE", "Year_month", "cases")],
            by=c("DANE", "Year_month"), all.x=TRUE)
df$cases[is.na(df$cases)] <- 0

df$year <- as.integer(substr(df$Year_month, 1, 4))

# Population already included in risk_monthly.csv.gz
pop_med <- aggregate(population ~ DANE, data=df, median, na.rm=TRUE)
names(pop_med)[2] <- "pop_med"
df <- merge(df, pop_med, by="DANE", all.x=TRUE)
df$population[is.na(df$population)] <- df$pop_med[is.na(df$population)]
df$pop_med <- NULL

# Incidence and outbreak
df$incidence_per_10k <- (df$cases / df$population) * 10000
df$has_outbreak <- as.integer(df$incidence_per_10k > 0)

# NOTE: risk_monthly.csv.gz ALREADY includes the columns potential_intro and R0_rel
# (saved in compute_final_risk.py from the original source files).
# Therefore NO additional merge with intro/r0 is needed.
# Only fill possible NAs.
df$potential_intro[is.na(df$potential_intro)] <- 0
df$R0_rel[is.na(df$R0_rel)] <- 0

# Sort
df <- df[order(df$DANE, df$Year_month), ]
rownames(df) <- NULL

cat(sprintf("  Merged: %d records, %d municipalities\n",
    nrow(df), length(unique(df$DANE))))
cat(sprintf("  Outbreak prevalence (all): %.2f%%\n", mean(df$has_outbreak) * 100))
cat(sprintf("  Outbreak prevalence >2000m: %.2f%%\n",
    mean(df$has_outbreak[df$altitude_min > 2000], na.rm=TRUE) * 100))

# --- 4. DLNM MODELS ---
cat("[4/6] Fitting DLNM models by municipality...\n")

municipios <- unique(df$DANE)
n_muni <- length(municipios)
cat(sprintf("  Using ALL %d municipalities for training\n", n_muni))

df$dlnm_risk <- NA
lag_importance <- data.frame(DANE=integer(), lag=integer(),
                             coef=numeric(), se=numeric(),
                             stringsAsFactors=FALSE)

set.seed(42)

# Separate model per municipality (each municipality has ~216 months)
for (i in seq_along(municipios)) {
  muni <- municipios[i]
  sub <- df[df$DANE == muni, ]
  sub <- sub[order(sub$Year_month), ]
  n_obs <- nrow(sub)

  if (n_obs < 15) {
    # Municipality with few data: use current risk as prediction
    df$dlnm_risk[df$DANE == muni] <- sub$risk
    next
  }

  sub$time <- 1:n_obs

  # Crossbasis for potential_intro (lag 0-3) with ns
  cb_intro <- tryCatch(
    crossbasis(sub$potential_intro, lag=3,
               argvar=list(fun="ns", knots=2),
               arglag=list(fun="ns", knots=2)),
    error=function(e) NULL
  )

  # Crossbasis for R0_rel (lag 0-3)
  cb_r0 <- tryCatch(
    crossbasis(sub$R0_rel, lag=3,
               argvar=list(fun="ns", knots=2),
               arglag=list(fun="ns", knots=2)),
    error=function(e) NULL
  )

  if (is.null(cb_intro) || is.null(cb_r0)) {
    df$dlnm_risk[df$DANE == muni] <- sub$risk
    next
  }

  tryCatch({
    # GLM binomial with smooth time trend (df=3 to avoid overfitting)
    model <- glm(
      has_outbreak ~ cb_intro + cb_r0 + ns(time, df=3),
      data=sub, family=binomial(link="logit"),
      control=glm.control(maxit=100)
    )

    preds <- predict(model, type="response")
    df$dlnm_risk[df$DANE == muni] <- preds

    # Extract lag importance for potential_intro
    red_intro <- tryCatch(crossreduce(cb_intro, model, value="lag"),
                          error=function(e) NULL)
    if (!is.null(red_intro)) {
      coefs <- coef(red_intro)
      se_vec <- sqrt(diag(vcov(red_intro)))
      for (lag_idx in seq_along(coefs)) {
        lag_importance <- rbind(lag_importance, data.frame(
          DANE=muni, lag=lag_idx - 1,
          coef=coefs[lag_idx], se=se_vec[lag_idx],
          component="potential_intro", stringsAsFactors=FALSE))
      }
    }

    # Extract lag importance for R0_rel
    red_r0 <- tryCatch(crossreduce(cb_r0, model, value="lag"),
                       error=function(e) NULL)
    if (!is.null(red_r0)) {
      coefs <- coef(red_r0)
      se_vec <- sqrt(diag(vcov(red_r0)))
      for (lag_idx in seq_along(coefs)) {
        lag_importance <- rbind(lag_importance, data.frame(
          DANE=muni, lag=lag_idx - 1,
          coef=coefs[lag_idx], se=se_vec[lag_idx],
          component="R0_rel", stringsAsFactors=FALSE))
      }
    }

  }, error=function(e) {
    df$dlnm_risk[df$DANE == muni] <- sub$risk
  })

  if (i %% 100 == 0) cat(sprintf("    %d/%d municipalities processed\n", i, n_muni))
}
cat(sprintf("  %d/%d municipalities processed\n", n_muni, n_muni))

# --- 5. EVALUATE METRICS (ONLY >2000m) ---
cat("[5/6] Computing metrics (only >2000m)...\n")

# Filter to municipalities >2000m for evaluation
df_high <- df[!is.na(df$dlnm_risk) & df$altitude_min > 2000, ]
cat(sprintf("  Evaluation subset: %d records, %d municipalities\n",
    nrow(df_high), length(unique(df_high$DANE))))

y_true  <- df_high$has_outbreak
y_score <- df_high$dlnm_risk

metrics <- list()
metrics$eval_subset <- "municipalities > 2000m only"
metrics$n_records   <- nrow(df_high)
metrics$n_munis     <- length(unique(df_high$DANE))

# AUC-ROC
roc_obj <- tryCatch(roc(y_true, y_score), error=function(e) NULL)
metrics$auc_roc <- if (!is.null(roc_obj)) as.numeric(auc(roc_obj)) else NA
metrics$auc_roc_ci <- if (!is.null(roc_obj)) as.numeric(ci.auc(roc_obj)) else NA

# AUC-PR
if (has_PRROC) {
  pr <- tryCatch(
    PRROC::pr.curve(scores.class0=y_score[y_true==1],
                    scores.class1=y_score[y_true==0]),
    error=function(e) NULL
  )
  metrics$auc_pr <- if (!is.null(pr)) pr$auc.integral else NA
} else {
  metrics$auc_pr <- NA
}

metrics$prevalence <- mean(y_true)
if (!is.na(metrics$auc_pr) && metrics$prevalence > 0) {
  metrics$auc_pr_lift <- metrics$auc_pr / metrics$prevalence
} else {
  metrics$auc_pr_lift <- NA
}

# Spearman
sp <- cor.test(y_score, y_true, method="spearman", exact=FALSE)
metrics$spearman_rho <- as.numeric(sp$estimate)
metrics$spearman_p   <- sp$p.value

# Kendall
kt <- cor.test(y_score, y_true, method="kendall", exact=FALSE)
metrics$kendall_tau <- as.numeric(kt$estimate)

# Recall@k
for (k_pct in c(0.01, 0.05, 0.10, 0.20)) {
  k <- max(1, floor(k_pct * length(y_true)))
  top_k <- order(y_score, decreasing=TRUE)[1:k]
  recall_k <- sum(y_true[top_k]) / sum(y_true)
  metrics[[paste0("recall_top_", k_pct*100, "pct")]] <- recall_k
  metrics[[paste0("lift_top_", k_pct*100, "pct")]]   <- recall_k / k_pct
}

# Lag importance (average across municipalities, using all data)
if (nrow(lag_importance) > 0) {
  lag_summary <- aggregate(cbind(coef, se) ~ lag + component,
                           data=lag_importance,
                           FUN=function(x) c(mean=mean(x), se=sd(x)/sqrt(length(x))))
  metrics$lag_importance <- list(
    lag       = lag_summary$lag,
    component = lag_summary$component,
    coef_mean = lag_summary$coef[, "mean"],
    coef_se   = lag_summary$coef[, "se"]
  )

  # Best lag for each component
  for (comp in c("potential_intro", "R0_rel")) {
    comp_data <- lag_importance[lag_importance$component == comp, ]
    if (nrow(comp_data) > 0) {
      coef_by_lag <- tapply(abs(comp_data$coef), comp_data$lag, mean)
      best <- as.integer(names(which.max(coef_by_lag)))
      metrics[[paste0("best_lag_", comp)]] <- best
    }
  }
}

# Report results
cat(sprintf("  AUC-ROC = %.4f\n", metrics$auc_roc))
if (!is.na(metrics$auc_pr))
  cat(sprintf("  AUC-PR  = %.4f (lift = %.2fx)\n", metrics$auc_pr, metrics$auc_pr_lift))
cat(sprintf("  Spearman rho = %.4f\n", metrics$spearman_rho))
cat(sprintf("  Kendall tau  = %.4f\n", metrics$kendall_tau))
cat(sprintf("  Recall@10%%   = %.1f%%\n", metrics$recall_top_10pct * 100))
if (!is.null(metrics$best_lag_potential_intro))
  cat(sprintf("  Best lag for potential_intro: %d\n", metrics$best_lag_potential_intro))
if (!is.null(metrics$best_lag_R0_rel))
  cat(sprintf("  Best lag for R0_rel: %d\n", metrics$best_lag_R0_rel))

# --- 6. SAVE RESULTS ---
cat("[6/6] Saving results...\n")

results_path <- file.path(OUTPUT_DIR, "dlnm_results.json")
write_json(metrics, results_path, digits=6, pretty=TRUE, auto_unbox=TRUE)
cat(sprintf("  Results: %s\n", results_path))

# Predictions only for >2000m (for comparison with S1-S3)
preds_path <- file.path(OUTPUT_DIR, "dlnm_predictions_high.csv")
write.csv(df_high[, c("DANE", "Year_month", "altitude_min", "risk", "dlnm_risk", "has_outbreak", "potential_intro", "R0_rel")],
          preds_path, row.names=FALSE)
cat(sprintf("  Predictions (>2000m): %s\n", preds_path))

# Detailed lag importance
if (nrow(lag_importance) > 0) {
  lag_path <- file.path(OUTPUT_DIR, "dlnm_lag_importance.csv")
  write.csv(lag_importance, lag_path, row.names=FALSE)
  cat(sprintf("  Lag importance: %s\n", lag_path))
}

cat("\n", rep("=", 70), "\n", sep="")
cat("DLNM STRATEGY COMPLETED\n")
cat("\nTo compare all 4 strategies, run:\n")
cat("  python lags/test_lags.py\n")
cat("(the script will auto-detect dlnm_results.json)\n")
cat(rep("=", 70), "\n", sep="")
