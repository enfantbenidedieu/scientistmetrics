# -*- coding: utf-8 -*-
from __future__ import annotations
from .association import association
from .powerset import powersetmodel

from .performance import (
    lag,
    diff,
    extractAIC,
    extractAICC,
    extractBIC,
    coefficients,
    logLik,
    LikelihoodRatioTest,
    HosmerLemeshowTest,
    MannWhitneyTest,
    residuals,
    rstandard,
    rstudent,
    explained_variance_score,
    r2_score,
    mean_squared_error,
    max_error,
    mean_absolute_error,
    median_absolute_error,
    r2_efron,
    r2_mcfadden,
    r2_mckelvey,
    r2_count,
    r2_count_adj,
    r2_coxsnell,
    r2_nagelkerke,
    r2_tjur,
    r2_kullback,
    r2_somers,
    r2_xu,
    r2,
    accuracy_score,
    error_rate,
    balanced_accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    multiclass_roc,
    check_autocorrelation,
    check_heteroscedasticity,
    check_model,
    check_normality,
    check_overdispersion,
    check_sphericity_bartlett,
    check_symmetric,
    model_performance,
    compare_performance,
    ggroc
)

__version__ = "0.0.4"
__name__ = "discrimintools"
__author__ = 'Duvérier DJIFACK ZEBAZE'
__email__ = 'duverierdjifack@gmail.com' 
