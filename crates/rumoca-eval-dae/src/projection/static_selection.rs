//! Exact branch selection for indices bound by the current structured point.

use super::*;

impl<'dae> Projection<'_, 'dae> {
    pub(super) fn static_selected_branch(
        &mut self,
        operands: dae::ExpressionOperands<'dae>,
    ) -> Result<dae::ExprId<'dae>, ProjectionError> {
        for ordinal in (0..operands.len() - 1).step_by(2) {
            if self.static_boolean(operands.get(ordinal).unwrap())? {
                return Ok(operands.get(ordinal + 1).unwrap());
            }
        }
        Ok(operands.get(operands.len() - 1).unwrap())
    }

    fn static_boolean(&mut self, expression: dae::ExprId<'dae>) -> Result<bool, ProjectionError> {
        let raw = expression.index();
        if !self.integer_stack.insert(raw) {
            return Err(ProjectionError::DynamicSubscript {
                span: self.node(expression).provenance().span(),
            });
        }
        let result = self.static_boolean_inner(expression);
        self.integer_stack.remove(&raw);
        result
    }

    fn static_boolean_inner(
        &mut self,
        expression: dae::ExprId<'dae>,
    ) -> Result<bool, ProjectionError> {
        use dae::BinaryOperator as B;
        let node = self.node(expression);
        let span = node.provenance().span();
        match node.operation() {
            dae::ExpressionOperation::Literal(dae::DaeLiteral::Boolean(value)) => Ok(*value),
            dae::ExpressionOperation::Unary {
                operator: dae::UnaryOperator::Not,
                operand,
            } => self.static_boolean(operand).map(|value| !value),
            dae::ExpressionOperation::Binary {
                operator: B::And,
                lhs,
                rhs,
            } => {
                let lhs = self.static_boolean(lhs)?;
                let rhs = self.static_boolean(rhs)?;
                Ok(lhs && rhs)
            }
            dae::ExpressionOperation::Binary {
                operator: B::Or,
                lhs,
                rhs,
            } => {
                let lhs = self.static_boolean(lhs)?;
                let rhs = self.static_boolean(rhs)?;
                Ok(lhs || rhs)
            }
            dae::ExpressionOperation::Binary { operator, lhs, rhs } => {
                let lhs = self.integer(lhs, 0)?;
                let rhs = self.integer(rhs, 0)?;
                match operator {
                    B::Equal => Ok(lhs == rhs),
                    B::NotEqual => Ok(lhs != rhs),
                    B::Less => Ok(lhs < rhs),
                    B::LessEqual => Ok(lhs <= rhs),
                    B::Greater => Ok(lhs > rhs),
                    B::GreaterEqual => Ok(lhs >= rhs),
                    _ => Err(ProjectionError::DynamicSubscript { span }),
                }
            }
            dae::ExpressionOperation::Coordinate(dae::CoordinateView::Parameter(parameter)) => {
                let binding = self
                    .view
                    .variable(parameter.into())
                    .unwrap()
                    .binding()
                    .ok_or(ProjectionError::DynamicSubscript { span })?;
                self.static_boolean(binding)
            }
            dae::ExpressionOperation::Conditional(operands) => {
                let selected = self.static_selected_branch(operands)?;
                self.static_boolean(selected)
            }
            _ => Err(ProjectionError::DynamicSubscript { span }),
        }
    }
}
