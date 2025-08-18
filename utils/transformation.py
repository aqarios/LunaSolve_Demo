"""
Binary constraint to penalty transformation for luna_quantum models.

This module provides functionality to transform binary optimization models
with inequality constraints into penalty form, making them suitable for
quantum annealing and other penalty-based solvers.
"""

import math
from typing import Tuple, Optional

from luna_quantum.decorators import transform
from luna_quantum.transformations import ActionType, PassManager, MaxBiasAnalysis
from luna_quantum import (
    Model,
    Variable,
    Vtype,
    Expression,
    Constant,
    HigherOrder,
    Linear,
    Quadratic,
    Comparator,
)


class BinaryConstraintError(ValueError):
    """Raised when a model contains non-binary variables during transformation."""

    pass


class UnsupportedConstraintError(NotImplementedError):
    """Raised when a constraint type is not yet supported."""

    pass


def constraint_back(solution, cache) -> None:
    """
    Backward transformation - converts penalty solutions back to constrained form.

    Args:
        solution: The solution from the penalty-transformed model
        cache: Transformation cache containing metadata

    Raises:
        NotImplementedError: This feature is not yet implemented
    """
    raise NotImplementedError("Backward transformation is not yet implemented")


@transform(backwards=constraint_back)
def binary_constraint_to_penalty_transform(
    model: Model, cache
) -> Tuple[Model, ActionType, Optional[dict]]:
    """
    Transform a binary model with inequality constraints to penalty form.

    This transformation converts a constrained binary optimization problem into
    an unconstrained penalty formulation suitable for quantum annealing. The
    process involves:

    1. Validating all variables are binary
    2. Calculating appropriate penalty factors
    3. Adding slack variables for inequality constraints using log2 encoding
    4. Converting constraints to penalty terms in the objective function

    Args:
        model: Input binary model with constraints to transform
        cache: Transformation cache containing analysis results

    Returns:
        Tuple containing:
        - Transformed model with penalty formulation
        - Action type indicating transformation status
        - Optional metadata dictionary

    Raises:
        BinaryConstraintError: If model contains non-binary variables
        UnsupportedConstraintError: If unsupported constraint types are found
    """
    # Step 1: Validate all variables are binary
    _validate_binary_model(model)

    # Step 2: Calculate penalty factor from objective coefficients
    penalty_factor = _calculate_penalty_factor(cache)

    # Step 3: Create new penalty-based model
    penalty_model = _create_penalty_model(model, penalty_factor)

    return penalty_model, ActionType.DidTransform, None


def _validate_binary_model(model: Model) -> None:
    """
    Validate that all variables in the model are binary.

    Args:
        model: Model to validate

    Raises:
        BinaryConstraintError: If any variable is not binary
    """
    non_binary_vars = [
        var.name for var in model.variables() if var.vtype != Vtype.Binary
    ]

    if non_binary_vars:
        raise BinaryConstraintError(
            f"Variables {non_binary_vars} are not binary. "
            "Only complete binary models can be converted."
        )


def _calculate_penalty_factor(cache) -> float:
    """
    Calculate the penalty factor from the maximum objective coefficient.

    The penalty factor is set to 10 times the maximum objective coefficient
    to ensure constraint violations are heavily penalized.

    Args:
        cache: Transformation cache containing MaxBiasAnalysis results

    Returns:
        Calculated penalty factor
    """
    max_objective_coeff = cache[MaxBiasAnalysis().name]
    return max_objective_coeff.val * 10


def _create_penalty_model(model: Model, penalty_factor: float) -> Model:
    """
    Create the penalty-based model from the original constrained model.

    Args:
        model: Original constrained model
        penalty_factor: Penalty factor for constraint violations

    Returns:
        New penalty-based model
    """
    penalty_model = Model(f"Penalty_Transform_{model.name}")

    with penalty_model.environment:
        # Copy original variables and objective
        var_mapping = _copy_variables(model, penalty_model)
        _copy_objective(model, penalty_model, var_mapping)

        # Process constraints and add penalty terms
        _process_constraints(model, penalty_model, var_mapping, penalty_factor)

    return penalty_model


def _copy_variables(model: Model, new_model: Model) -> dict:
    """
    Copy all variables from the original model to the new model.

    Args:
        model: Original model
        new_model: Target model

    Returns:
        Dictionary mapping original variable names to new variables
    """
    var_mapping = {}
    for var in model.variables():
        new_var = Variable(var.name, vtype=var.vtype)
        var_mapping[var.name] = new_var
    return var_mapping


def _copy_objective(model: Model, new_model: Model, var_mapping: dict) -> None:
    """
    Copy the objective function from the original model to the new model.

    Args:
        model: Original model
        new_model: Target model
        var_mapping: Mapping from original to new variables

    Raises:
        UnsupportedConstraintError: For unsupported objective term types
    """
    for vars_term, bias in model.objective.items():
        match vars_term:
            case Constant():
                new_model.objective += bias
            case Linear(x):
                new_model.objective += bias * var_mapping[x.name]
            case Quadratic(x, y):
                new_model.objective += bias * var_mapping[x.name] * var_mapping[y.name]
            case HigherOrder():
                raise UnsupportedConstraintError(
                    "Higher-order objective terms are not yet supported"
                )


def _process_constraints(
    model: Model, new_model: Model, var_mapping: dict, penalty_factor: float
) -> None:
    """
    Process all constraints and add corresponding penalty terms.

    Args:
        model: Original model with constraints
        new_model: Target penalty model
        var_mapping: Variable mapping
        penalty_factor: Penalty factor for violations
    """
    for i, constraint in enumerate(model.constraints):
        if constraint.comparator == Comparator.Eq and constraint.rhs == 1:
            _add_equality_penalty(constraint, new_model, var_mapping, penalty_factor)
        elif constraint.comparator == Comparator.Le:
            _add_inequality_penalty(
                constraint, new_model, var_mapping, penalty_factor, i
            )
        else:
            raise UnsupportedConstraintError(
                f"Constraint type {constraint.comparator} is not yet supported"
            )


def _add_equality_penalty(
    constraint, new_model: Model, var_mapping: dict, penalty_factor: float
) -> None:
    """
    Add penalty term for equality constraint (lhs = 1).

    Args:
        constraint: Equality constraint to process
        new_model: Target penalty model
        var_mapping: Variable mapping
        penalty_factor: Penalty factor

    Raises:
        UnsupportedConstraintError: For unsupported constraint term types
    """
    new_lhs = Expression()

    for vars_term, bias in constraint.lhs.items():
        match vars_term:
            case Constant():
                raise UnsupportedConstraintError(
                    "Constant terms in equality constraints are not yet supported"
                )
            case Linear(x):
                if bias != 1.0:
                    raise UnsupportedConstraintError(
                        f"Non-unit coefficients ({bias}) in equality constraints "
                        "are not yet supported"
                    )
                new_lhs += var_mapping[x.name]
            case Quadratic() | HigherOrder():
                raise UnsupportedConstraintError(
                    f"{type(vars_term).__name__} terms in equality constraints "
                    "are not yet supported"
                )

    # Add penalty term: penalty_factor * (lhs - 1)^2
    penalty_term = penalty_factor * (new_lhs - 1) ** 2
    new_model.objective += penalty_term


def _add_inequality_penalty(
    constraint,
    new_model: Model,
    var_mapping: dict,
    penalty_factor: float,
    constraint_index: int,
) -> None:
    """
    Add penalty term for inequality constraint (lhs <= rhs).

    Args:
        constraint: Inequality constraint to process
        new_model: Target penalty model
        var_mapping: Variable mapping
        penalty_factor: Penalty factor
        constraint_index: Index for slack variable naming

    Raises:
        UnsupportedConstraintError: For unsupported constraint term types
    """
    lhs = constraint.lhs
    rhs = constraint.rhs
    rhs_constant = rhs - lhs.get_offset()

    # Calculate minimum LHS value and slack range
    slack_range = _calculate_slack_range(lhs, rhs_constant)

    # Build new LHS expression
    new_lhs = _build_lhs_expression(lhs, var_mapping)

    # Add slack variables using log2 encoding
    if slack_range > 0:
        _add_slack_variables(new_lhs, slack_range, constraint_index)

    # Add penalty term: penalty_factor * (rhs - lhs)^2
    penalty_term = penalty_factor * (rhs - new_lhs) ** 2
    new_model.objective += penalty_term


def _calculate_slack_range(lhs: Expression, rhs_constant: float) -> int:
    """
    Calculate the range needed for slack variables.

    Args:
        lhs: Left-hand side expression
        rhs_constant: Right-hand side constant

    Returns:
        Required slack range
    """
    # Find minimum LHS value by setting negative coefficients to 1
    min_contribution = sum(
        bias
        for vars_term, bias in lhs.items()
        if isinstance(vars_term, Linear) and bias < 0
    )

    return int(rhs_constant - min_contribution)


def _build_lhs_expression(lhs: Expression, var_mapping: dict) -> Expression:
    """
    Build new LHS expression with mapped variables.

    Args:
        lhs: Original LHS expression
        var_mapping: Variable mapping

    Returns:
        New LHS expression with mapped variables

    Raises:
        UnsupportedConstraintError: For unsupported term types
    """
    new_lhs = Expression()

    for vars_term, bias in lhs.items():
        match vars_term:
            case Constant():
                new_lhs += bias
            case Linear(x):
                new_lhs += bias * var_mapping[x.name]
            case Quadratic(x, y):
                new_lhs += bias * var_mapping[x.name] * var_mapping[y.name]
            case HigherOrder():
                raise UnsupportedConstraintError(
                    "Higher-order terms in inequality constraints are not yet supported"
                )

    return new_lhs


def _add_slack_variables(
    lhs_expression: Expression, slack_range: int, constraint_index: int
) -> None:
    """
    Add slack variables using log2 encoding to the LHS expression.

    Args:
        lhs_expression: Expression to add slack variables to
        slack_range: Range of slack values needed
        constraint_index: Index for variable naming
    """
    if slack_range <= 0:
        return

    num_slack_bits = math.ceil(math.log2(slack_range + 1))

    for bit_index in range(num_slack_bits):
        slack_var = Variable(
            f"slack_{constraint_index}_{bit_index}", vtype=Vtype.Binary
        )
        lhs_expression += slack_var * (2**bit_index)


def move_constraints(model: Model) -> Model:
    """
    Transform a luna_quantum model with inequality constraints to penalty form.

    This is the main entry point for the constraint-to-penalty transformation.
    It sets up the transformation pipeline and executes the conversion.

    Args:
        model: Input luna_quantum Model with inequality constraints

    Returns:
        Transformed Model with penalty terms instead of inequality constraints

    Raises:
        BinaryConstraintError: If model contains non-binary variables
        UnsupportedConstraintError: If unsupported constraint types are found
    """
    pass_manager = PassManager(
        passes=[MaxBiasAnalysis(), binary_constraint_to_penalty_transform]
    )

    transformation_result = pass_manager.run(model=model)
    return transformation_result.model
