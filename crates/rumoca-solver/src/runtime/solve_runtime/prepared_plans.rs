//! Fixed memberships cannot enter execution without their checked selections.

use super::plans::{
    RootConditionPlan, RootConditionPlanEntry, VisibleExpressionGroup, VisibleValuePlan,
    VisibleValuePlanEntry,
};
use super::prepared_outputs::PreparedOutputSelection;
use super::*;

#[derive(Clone)]
pub(super) struct PreparedRootConditionPlan {
    pub(super) entries: Vec<RootConditionPlanEntry>,
    pub(super) evaluated: PreparedOutputSelection,
    pub(super) search: PreparedOutputSelection,
}

impl RootConditionPlan {
    pub(super) fn prepare(
        self,
        source: &PreparedScalarProgramBlock,
        backend: Option<&Rc<dyn SolveExecutionBackend>>,
        primary: Option<&PreparedRootConditionPlan>,
    ) -> Result<PreparedRootConditionPlan, EvalSolveError> {
        let evaluated = PreparedOutputSelection::new(
            source,
            &self.evaluated_rows,
            backend,
            primary.map(|plan| &plan.evaluated),
        )?;
        let search = if self.evaluated_rows == self.search_rows {
            evaluated.clone()
        } else {
            PreparedOutputSelection::new(
                source,
                &self.search_rows,
                backend,
                primary.map(|plan| &plan.search),
            )?
        };
        Ok(PreparedRootConditionPlan {
            entries: self.entries,
            evaluated,
            search,
        })
    }
}

#[derive(Clone)]
pub(super) struct PreparedVisibleValuePlan {
    pub(super) entries: Vec<VisibleValuePlanEntry>,
    pub(super) expression_groups: Vec<VisibleExpressionGroup>,
    pub(super) expressions: PreparedOutputSelection,
}

impl VisibleValuePlan {
    pub(super) fn prepare(
        self,
        source: &PreparedScalarProgramBlock,
        backend: Option<&Rc<dyn SolveExecutionBackend>>,
        primary: Option<&PreparedVisibleValuePlan>,
    ) -> Result<PreparedVisibleValuePlan, EvalSolveError> {
        let expressions = PreparedOutputSelection::new(
            source,
            &self.expression_rows,
            backend,
            primary.map(|plan| &plan.expressions),
        )?;
        Ok(PreparedVisibleValuePlan {
            entries: self.entries,
            expression_groups: self.expression_groups,
            expressions,
        })
    }
}
