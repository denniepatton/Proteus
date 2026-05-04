from __future__ import annotations

from typing import Dict, List, Tuple


SECTION_OUTPUT_DIRS: Dict[str, str] = {
    "Examples": "Examples",
    "Benchmarking_Synthetic": "Benchmarking_Synthetic",
    "Benchmarking_Patient": "Benchmarking_Patient",
    "Calibration": "Calibration",
    "Validation_PDX": "Validation_PDX",
    "Validation_Patient": "Validation_Patient",
    "Discovery_Patient": "Discovery_Patient",
}


BENCHMARK_COMPLETE_MODELS: List[str] = [
    "Proteus",
    "Proteus_NoSeq",
    "Proteus_OnlySeq",
    "XGBoost",
    "EPIC_GEP",
    "EPIC_NDR",
    "EPIC_PFE",
    "Naive",
]


BENCHMARK_SUBSET_MODELS: List[str] = [
    "Proteus",
    "XGBoost",
    "EPIC_GEP",
    "Naive",
]


SYNTHETIC_PROTEUS_MODEL_PATHS: Dict[str, str] = {
    "Proteus": "results_holdout/BioDVN/PHS/predictions_tumor.tsv",
    "Proteus_NoSeq": "results_holdout/BioDVN_NoSeq/PHS/predictions_tumor.tsv",
    "Proteus_OnlySeq": "results_holdout/BioDVN_OnlySeq/PHS/predictions_tumor.tsv",
}


SYNTHETIC_XGB_PATH: str = "results_holdout/XGBoost/PHS/predictions_tumor.tsv"
SYNTHETIC_EPIC_PATH: str = "results_holdout/EPIC-Seq/PHS/EPICSeqInferredExpressionValues.txt"


PATIENT_BENCHMARK_COHORTS: List[str] = ["CRPC-WCDT", "BLCA-RA"]


PATIENT_PROTEUS_MODEL_PATHS: Dict[str, Dict[str, str]] = {
    "CRPC-WCDT": {
        "Proteus": "results_holdout/BioDVN/CRPC-WCDT/predictions_tumor.tsv",
        "Proteus_NoSeq": "results_holdout/BioDVN_NoSeq/CRPC-WCDT/predictions_tumor.tsv",
        "Proteus_OnlySeq": "results_holdout/BioDVN_OnlySeq/CRPC-WCDT/predictions_tumor.tsv",
    },
    "BLCA-RA": {
        "Proteus": "results_holdout/BioDVN/BLCA-RA/predictions_tumor.tsv",
        "Proteus_NoSeq": "results_holdout/BioDVN_NoSeq/BLCA-RA/predictions_tumor.tsv",
        "Proteus_OnlySeq": "results_holdout/BioDVN_OnlySeq/BLCA-RA/predictions_tumor.tsv",
    },
}


PATIENT_XGB_PATHS: Dict[str, str] = {
    "CRPC-WCDT": "results_holdout/XGBoost/CRPC-WCDT/predictions_tumor.tsv",
    "BLCA-RA": "results_holdout/XGBoost/BLCA-RA/predictions_tumor.tsv",
}


PATIENT_EPIC_PATHS: Dict[str, str] = {
    "CRPC-WCDT": "results_holdout/EPIC-Seq/CRPC-WCDT/EPICSeqInferredExpressionValues.txt",
    "BLCA-RA": "results_holdout/EPIC-Seq/BLCA-RA/EPICSeqInferredExpressionValues.txt",
}


VALIDATION_PATIENT_COHORTS: List[str] = [
    "CRPC-WCDT",
    "BLCA-RA",
    "CRPC-RA",
    "SCLC",
]


VALIDATION_PATIENT_HOLDOUT_COHORTS: List[str] = ["CRPC-WCDT", "BLCA-RA"]
VALIDATION_PATIENT_CV_COHORTS: List[str] = ["CRPC-RA", "SCLC"]
VALIDATION_PDX_COHORTS: List[str] = ["PC-PDX", "SCLC-PDX"]


VALIDATION_PATIENT_HOLDOUT_PATH_TEMPLATE: str = (
    "results_holdout/BioDVN/{cohort}/predictions_tumor.tsv"
)
VALIDATION_PATIENT_CV_RESULTS_DIR: str = "results_training/proteus_plca-model_cv"
VALIDATION_PDX_CV_RESULTS_DIR: str = "results_training/proteus_base-model_cv"


VALIDATION_PDX_HOLDOUT_PATHS: Dict[str, str] = {
    "PC-PDX": "results_holdout/BioDVN/PC-PDX/predictions_tumor.tsv",
    "SCLC-PDX": "results_holdout/BioDVN/SCLC-PDX/predictions_tumor.tsv",
}


DISCOVERY_COHORTS: List[Tuple[str, str | None]] = [
    ("CRPC-Pfizer", None),
    ("CRPC-Pluvicto", "CRPC-Pluvicto"),
    ("CRPC-UW", "CRPC-UW"),
    ("BC-Ghana", None),
]


DISCOVERY_MODEL_PATHS: List[Tuple[str, str]] = [
    ("Proteus", "results_holdout/BioDVN/{cohort}/predictions_tumor.tsv"),
]


DISCOVERY_CROSS_COHORT_DIRNAME: str = "CrossCohort"


EPIC_SCORE_COLUMNS: List[str] = ["inferredGEP", "PFE", "NDR_relative_2k"]


EPIC_DISPLAY_NAMES: Dict[str, str] = {
    "inferredGEP": "EPIC_GEP",
    "PFE": "EPIC_PFE",
    "NDR_relative_2k": "EPIC_NDR",
}


def benchmark_model_views() -> Dict[str, List[str]]:
    return {
        "COMPLETE": BENCHMARK_COMPLETE_MODELS,
        "SUBSET": BENCHMARK_SUBSET_MODELS,
    }