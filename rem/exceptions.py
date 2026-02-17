"""
REM SDK Exceptions

All exceptions inherit from REMError for easy catch-all handling.
"""


class REMError(Exception):
    """Base exception for all REM SDK errors."""

    def __init__(self, message: str, status_code: int = 0, error_code: str = ""):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class AuthenticationError(REMError):
    """Raised when API key is invalid, expired, or revoked."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401, error_code="UNAUTHORIZED")


class NotFoundError(REMError):
    """Raised when a resource (collection, vector) is not found."""

    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404, error_code="NOT_FOUND")


class QuotaExceededError(REMError):
    """Raised when namespace quota is exceeded (vectors, queries, collections)."""

    def __init__(self, message: str = "Quota exceeded"):
        super().__init__(message, status_code=429, error_code="QUOTA_EXCEEDED")


class ValidationError(REMError):
    """Raised when request parameters are invalid (dimension mismatch, bad name, etc)."""

    def __init__(self, message: str = "Invalid parameters"):
        super().__init__(message, status_code=400, error_code="VALIDATION_ERROR")


class ServerError(REMError):
    """Raised when the server returns a 5xx error."""

    def __init__(self, message: str = "Server error"):
        super().__init__(message, status_code=500, error_code="SERVER_ERROR")


class REMTimeoutError(REMError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(message, status_code=0, error_code="TIMEOUT")


# Alias for backwards compatibility (but prefer REMTimeoutError to avoid shadowing builtin)
TimeoutError = REMTimeoutError
