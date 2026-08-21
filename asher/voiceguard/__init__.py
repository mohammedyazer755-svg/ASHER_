"""Asher VoiceGuard: consented recordings and honestly calibrated speaker access.

The package intentionally imports no microphone, Torch, SpeechBrain, NumPy, or
scikit-learn modules at import time. Live capture and optional pretrained
embeddings are loaded only when their APIs are called.
"""

from .audio import PcmAudio, read_wav, sha256_file, write_wav
from .augmentation import AugmentationConfig, augment_audio, augment_session, augment_wav
from .dataset import (
    DatasetSample,
    DatasetSplit,
    FeatureExample,
    VoiceDataset,
    load_dataset,
    session_separated_split,
)
from .enrollment import EnrollmentManager, EnrollmentRecord, RevocationResult
from .exceptions import (
    AudioFormatError,
    DatasetError,
    EnrollmentError,
    ManifestError,
    MLDependencyError,
    ModelError,
    RecordingUnavailableError,
    VoiceGuardError,
)
from .features import (
    FeatureExtractor,
    FeatureExtractorMetadata,
    PretrainedEmbeddingAdapter,
    SpeechBrainECAPAAdapter,
    StatisticalFeatureExtractor,
)
from .metrics import (
    ConditionMetrics,
    EvaluationObservation,
    EvaluationReport,
    evaluate_predictions,
    observations_from_scores,
)
from .model import CalibratedVoiceGuardModel, VerificationResult
from .recording import RecordingSession, load_manifest
from .schema import (
    SampleCondition,
    SampleOrigin,
    SampleRecord,
    SessionManifest,
    SpeakerRole,
    TrainingTask,
)
from .training import (
    TrainingConfig,
    TrainingResult,
    VoiceGuardTrainer,
    feature_examples_from_dataset,
    ml_dependencies_available,
    prepare_positive_negative_examples,
    prepare_training_examples,
    train_from_feature_examples,
    train_speaker_classifier,
    train_wake_verifier,
)

__all__ = [
    "AudioFormatError",
    "AugmentationConfig",
    "CalibratedVoiceGuardModel",
    "ConditionMetrics",
    "DatasetError",
    "DatasetSample",
    "DatasetSplit",
    "EnrollmentError",
    "EnrollmentManager",
    "EnrollmentRecord",
    "EvaluationObservation",
    "EvaluationReport",
    "FeatureExample",
    "FeatureExtractor",
    "FeatureExtractorMetadata",
    "ManifestError",
    "MLDependencyError",
    "ModelError",
    "PcmAudio",
    "PretrainedEmbeddingAdapter",
    "RecordingSession",
    "RecordingUnavailableError",
    "RevocationResult",
    "SampleCondition",
    "SampleOrigin",
    "SampleRecord",
    "SessionManifest",
    "SpeechBrainECAPAAdapter",
    "SpeakerRole",
    "StatisticalFeatureExtractor",
    "TrainingConfig",
    "TrainingResult",
    "TrainingTask",
    "VerificationResult",
    "VoiceDataset",
    "VoiceGuardError",
    "VoiceGuardTrainer",
    "augment_audio",
    "augment_session",
    "augment_wav",
    "feature_examples_from_dataset",
    "load_dataset",
    "load_manifest",
    "ml_dependencies_available",
    "observations_from_scores",
    "prepare_positive_negative_examples",
    "prepare_training_examples",
    "read_wav",
    "session_separated_split",
    "sha256_file",
    "train_from_feature_examples",
    "train_speaker_classifier",
    "train_wake_verifier",
    "write_wav",
    "evaluate_predictions",
]
