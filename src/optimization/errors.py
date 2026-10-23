"""Domain errors shared by optimizer validation and its HTTP boundary."""
from collections.abc import Mapping
import math


class OptimizationInputError(ValueError):
    """Portfolio data or requested constraints cannot define a valid problem."""


PortfolioValidationError = OptimizationInputError


class ConstraintFeasibilityError(OptimizationInputError):
    """The requested long-only allocation is invalid or infeasible."""


def validate_targets(targets, tol=1e-9):
    if not isinstance(targets, Mapping) or not targets:
        raise ConstraintFeasibilityError("Block targets must be a nonempty mapping.")
    values = []
    for name, value in targets.items():
        if not isinstance(name, str) or not name.strip():
            raise ConstraintFeasibilityError("Block names must be nonempty strings.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ConstraintFeasibilityError(f"Target for {name} must be a finite number in [0, 1].") from exc
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ConstraintFeasibilityError(f"Target for {name} must be finite and in [0, 1].")
        values.append(number)
    total = math.fsum(values)
    if abs(total - 1.) > tol:
        raise ConstraintFeasibilityError(f"Block targets must sum to 1.0, got {total}.")
