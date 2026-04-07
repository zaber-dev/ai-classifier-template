class AIClassifierError(Exception):
    """Base exception for this package."""


class ConfigurationError(AIClassifierError):
    """Raised when the pipeline configuration is invalid."""


class DataFormatError(AIClassifierError):
    """Raised when input data is malformed or unusable."""


class ModelArtifactError(AIClassifierError):
    """Raised when model artifact read/write operations fail."""
