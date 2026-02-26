# Interpretation module
# 
# WARNING: This module contains legacy TensorFlow code and needs to be 
# rewritten for PyTorch. The following functions still use TensorFlow:
# - load_best_model() - uses tf.keras.models.load_model
# - get_data() - uses TFRecord parsing
# - shap_analysis() - uses TensorFlow-based SHAP
#
# TODO: Port to PyTorch using Captum for interpretability
#
# For now, importing will fail unless TensorFlow is installed.

try:
    from .interpretation_utils import *
except ImportError as e:
    import warnings
    warnings.warn(
        f"Interpretation module not available: {e}. "
        "This module requires TensorFlow which is not installed. "
        "A PyTorch-native version using Captum is planned."
    )
