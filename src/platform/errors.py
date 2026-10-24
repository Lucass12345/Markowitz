class PlatformError(ValueError):
    status_code = 422

class NotFoundError(PlatformError):
    status_code = 404

class ConflictError(PlatformError):
    status_code = 409

class ServiceUnavailableError(RuntimeError):
    status_code = 503

class TransientError(ServiceUnavailableError):
    pass

class JobCancelled(Exception):
    pass

class LeaseLost(Exception):
    pass
