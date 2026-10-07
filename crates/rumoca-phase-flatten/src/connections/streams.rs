//! Stream connection semantics (MLS §15).
//!
//! Stream connection sets are built per connect() scope, like flow sets. A
//! member is an outside connector when it is an interface connector of the
//! scope model, and an inside connector otherwise. The mixing rule of MLS
//! §15.2 then gives:
//!
//! - STRM-004: the stream variable of every outside connector below the root
//!   model gets one equation, the mixture of what the other members of its
//!   set supply. Root-model outside connectors are the model interface
//!   (MLS §4.7) and receive their equation from the enclosing model.
//! - `inStream(v)` of an inside member is the mixture supplied by the other
//!   members of its set. An outside member supplies `inStream` of itself,
//!   which is resolved one scope up, where it is an inside member.
//! - `actualStream(v)` is `inStream(v)` while the connector's flow enters the
//!   component and `v` otherwise (MLS §15.3).
//!
//! A member that is not connected at a parent scope supplies its own stream
//! value. Mixing weights use `positiveMax(x) = max(x, STREAM_FLOW_EPSILON)`,
//! so the mixture is continuous across flow reversal and never divides by
//! zero (STRM-008, STRM-009). A set with exactly one other member resolves
//! to that member's value without weights.

use super::*;

/// Floor of `positiveMax` in the MLS §15.2 mixing weights, in the unit of the
/// connector's flow variable.
const STREAM_FLOW_EPSILON: f64 = 1.0e-10;

/// Stream variables joined by connect() equations declared in one scope.
pub(super) struct StreamConnectionSet {
    pub(super) scope: String,
    pub(super) members: Vec<rumoca_core::VarName>,
    pub(super) span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Inside,
    Outside,
}

struct Member {
    var: rumoca_core::VarName,
    role: Role,
}

struct ResolvedSet {
    scope: String,
    members: Vec<Member>,
    span: Span,
}

/// Stream connection sets with their MLS §15 roles and connector flows.
pub(super) struct StreamConnections {
    sets: Vec<ResolvedSet>,
    /// The set in which a stream variable is an inside member.
    inside_set_of: FxHashMap<rumoca_core::VarName, usize>,
    /// The flow variable of each connected stream variable's connector.
    flow_of: FxHashMap<rumoca_core::VarName, rumoca_core::VarName>,
}

impl StreamConnections {
    pub(super) fn new(
        flat: &flat::Model,
        sets: Vec<StreamConnectionSet>,
        interface_roots: &InterfaceConnectorRootsByScope,
    ) -> Result<Self, FlattenError> {
        let connector_flows = ConnectorFlows::new(flat);
        let mut connections = Self {
            sets: Vec::with_capacity(sets.len()),
            inside_set_of: FxHashMap::default(),
            flow_of: FxHashMap::default(),
        };
        for set in sets {
            connections.add_set(flat, set, interface_roots, &connector_flows)?;
        }
        Ok(connections)
    }

    fn add_set(
        &mut self,
        flat: &flat::Model,
        set: StreamConnectionSet,
        interface_roots: &InterfaceConnectorRootsByScope,
        connector_flows: &ConnectorFlows,
    ) -> Result<(), FlattenError> {
        let index = self.sets.len();
        let mut members = Vec::with_capacity(set.members.len());
        for var in set.members {
            let role = if is_interface_connection_path_for_scope(
                var.as_str(),
                &set.scope,
                interface_roots,
            ) {
                Role::Outside
            } else {
                Role::Inside
            };
            if role == Role::Inside
                && let Some(previous) = self.inside_set_of.insert(var.clone(), index)
            {
                return Err(FlattenError::internal(format!(
                    "stream variable `{}` is an inside connector in connection scopes `{}` and `{}`",
                    var.as_str(),
                    self.sets[previous].scope,
                    set.scope
                )));
            }
            // STRM-002 guarantees the flow in resolved models; a weight that
            // needs a missing flow reports it where the mixture is built.
            if let Some(flow) = connector_flows.of_stream(flat, &var) {
                self.flow_of.insert(var.clone(), flow);
            }
            members.push(Member { var, role });
        }
        self.sets.push(ResolvedSet {
            scope: set.scope,
            members,
            span: set.span,
        });
        Ok(())
    }

    pub(super) fn mark_connected(&self, flat: &mut flat::Model) {
        for set in &self.sets {
            for member in &set.members {
                mark_connected(flat, &member.var);
            }
        }
    }

    /// STRM-004: one mixing equation per outside connector below the root.
    pub(super) fn generate_outside_connector_equations(
        &self,
        flat: &mut flat::Model,
    ) -> Result<(), FlattenError> {
        for set in self.sets.iter().filter(|set| !set.scope.is_empty()) {
            let provenance = require_connection_provenance(set.span, "stream mixing equation")?;
            for member in set.members.iter().filter(|m| m.role == Role::Outside) {
                let mixture = self.mixture(set, &member.var, &[], provenance.span(), 0)?;
                let residual = create_equality_residual(
                    var_to_expr(&member.var, provenance),
                    mixture,
                    provenance,
                );
                let scalar_count = flat
                    .variables
                    .get(&member.var)
                    .map(compute_var_scalar_count)
                    .unwrap_or(1);
                let origin = rumoca_ir_flat::EquationOrigin::Connection {
                    lhs: member.var.as_str().to_string(),
                    rhs: format!("stream mixture in `{}`", set.scope),
                };
                flat.add_equation(flat::Equation::new_array(
                    residual,
                    set.span,
                    origin,
                    scalar_count,
                ));
            }
        }
        Ok(())
    }

    /// Replace `inStream` and `actualStream` with their connection-set values.
    pub(super) fn rewrite_stream_operators(
        &self,
        flat: &mut flat::Model,
    ) -> Result<(), FlattenError> {
        use rumoca_core::ExpressionRewriter;

        let mut rewriter = StreamOperatorRewriter {
            connections: self,
            error: None,
        };
        for eq in flat
            .equations
            .iter_mut()
            .chain(flat.initial_equations.iter_mut())
        {
            eq.residual = rewriter.rewrite_expression(&eq.residual);
        }
        for var in flat.variables.values_mut() {
            if let Some(binding) = var.binding.take() {
                var.binding = Some(rewriter.rewrite_expression(&binding));
            }
        }
        match rewriter.error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// `inStream(var)`: what the rest of `var`'s inside set supplies to it.
    fn in_stream(
        &self,
        var: &rumoca_core::VarName,
        subscripts: &[rumoca_core::Subscript],
        span: Span,
        depth: usize,
    ) -> Result<rumoca_core::Expression, FlattenError> {
        match self.inside_set_of.get(var) {
            Some(&index) => self.mixture(&self.sets[index], var, subscripts, span, depth),
            None => Ok(member_value(var, subscripts, span)),
        }
    }

    /// Mixture supplied to `receiver` by the other members of `set`.
    fn mixture(
        &self,
        set: &ResolvedSet,
        receiver: &rumoca_core::VarName,
        subscripts: &[rumoca_core::Subscript],
        span: Span,
        depth: usize,
    ) -> Result<rumoca_core::Expression, FlattenError> {
        if depth > self.sets.len() {
            return Err(FlattenError::internal(format!(
                "stream connection sets around `{}` form a cycle",
                receiver.as_str()
            )));
        }
        let suppliers: Vec<&Member> = set.members.iter().filter(|m| &m.var != receiver).collect();
        if let [only] = suppliers.as_slice() {
            return self.supplied_value(only, subscripts, span, depth);
        }
        let mut weighted = Vec::with_capacity(suppliers.len());
        let mut weights = Vec::with_capacity(suppliers.len());
        for member in suppliers {
            let weight = self.supply_weight(member, span)?;
            let value = self.supplied_value(member, subscripts, span, depth)?;
            weighted.push(binary(
                rumoca_core::OpBinary::Mul,
                weight.clone(),
                value,
                span,
            ));
            weights.push(weight);
        }
        Ok(binary(
            rumoca_core::OpBinary::Div,
            sum(weighted, span),
            sum(weights, span),
            span,
        ))
    }

    /// The stream value a member carries into its connection set.
    fn supplied_value(
        &self,
        member: &Member,
        subscripts: &[rumoca_core::Subscript],
        span: Span,
        depth: usize,
    ) -> Result<rumoca_core::Expression, FlattenError> {
        match member.role {
            Role::Inside => Ok(member_value(&member.var, subscripts, span)),
            Role::Outside => self.in_stream(&member.var, subscripts, span, depth + 1),
        }
    }

    /// `positiveMax` of the flow a member supplies into its connection set.
    fn supply_weight(
        &self,
        member: &Member,
        span: Span,
    ) -> Result<rumoca_core::Expression, FlattenError> {
        let flow = self.connector_flow(&member.var)?;
        let flow = member_value(flow, &[], span);
        // Inside flow is positive into the component, so its supply into the
        // set is the negated flow; outside flow is positive into the scope
        // model, which is a supply into the set.
        let supplied = match member.role {
            Role::Inside => rumoca_core::Expression::Unary {
                op: rumoca_core::OpUnary::Minus,
                rhs: Box::new(flow),
                span,
            },
            Role::Outside => flow,
        };
        Ok(rumoca_core::Expression::BuiltinCall {
            function: rumoca_core::BuiltinFunction::Max,
            args: vec![supplied, real_literal(STREAM_FLOW_EPSILON, span)],
            span,
        })
    }

    fn connector_flow(
        &self,
        var: &rumoca_core::VarName,
    ) -> Result<&rumoca_core::VarName, FlattenError> {
        self.flow_of
            .get(var)
            .ok_or_else(|| FlattenError::missing_flow_variable(var.as_str(), "<connector flow>"))
    }

    /// `actualStream(var)`: inflow carries `inStream(var)`, outflow `var`.
    fn actual_stream(
        &self,
        var: &rumoca_core::VarName,
        subscripts: &[rumoca_core::Subscript],
        span: Span,
    ) -> Result<rumoca_core::Expression, FlattenError> {
        if !self.flow_of.contains_key(var) {
            return Ok(member_value(var, subscripts, span));
        }
        let flow = member_value(self.connector_flow(var)?, &[], span);
        let entering = binary(
            rumoca_core::OpBinary::Gt,
            flow,
            real_literal(0.0, span),
            span,
        );
        Ok(rumoca_core::Expression::If {
            branches: vec![(entering, self.in_stream(var, subscripts, span, 0)?)],
            else_branch: Box::new(member_value(var, subscripts, span)),
            span,
        })
    }
}

/// Flow variables indexed by their structured connector reference.
///
/// MLS §15.1 places exactly one flow variable at the same connector level as
/// each stream variable, so a stream variable's flow is the single flow whose
/// reference shares the stream variable's connector prefix.
struct ConnectorFlows {
    by_connector: FxHashMap<rumoca_core::VarName, Vec<rumoca_core::VarName>>,
}

impl ConnectorFlows {
    fn new(flat: &flat::Model) -> Self {
        let mut by_connector: FxHashMap<rumoca_core::VarName, Vec<rumoca_core::VarName>> =
            FxHashMap::default();
        for (name, var) in flat.variables.iter().filter(|(_, var)| var.flow) {
            if let Some(connector) = var.component_ref.as_ref().and_then(connector_key) {
                by_connector
                    .entry(connector)
                    .or_default()
                    .push(name.clone());
            }
        }
        Self { by_connector }
    }

    fn of_stream(
        &self,
        flat: &flat::Model,
        stream: &rumoca_core::VarName,
    ) -> Option<rumoca_core::VarName> {
        let reference = flat.variables.get(stream)?.component_ref.as_ref()?;
        match self
            .by_connector
            .get(&connector_key(reference)?)?
            .as_slice()
        {
            [flow] => Some(flow.clone()),
            _ => None,
        }
    }
}

/// The connector that declares a connector member, from its structured parts.
fn connector_key(member: &rumoca_core::ComponentReference) -> Option<rumoca_core::VarName> {
    let parts = member.component_scope().prefix_parts();
    (!parts.is_empty()).then(|| {
        rumoca_core::ComponentReference {
            local: member.local,
            span: member.span,
            parts: parts.to_vec(),
            def_id: None,
        }
        .to_var_name()
    })
}

struct StreamOperatorRewriter<'a> {
    connections: &'a StreamConnections,
    error: Option<FlattenError>,
}

impl StreamOperatorRewriter<'_> {
    /// The connection-set value of a stream operator call, or `None` when
    /// `expr` is not `inStream`/`actualStream` of a variable reference.
    fn resolve(
        &self,
        expr: &rumoca_core::Expression,
    ) -> Option<Result<rumoca_core::Expression, FlattenError>> {
        let rumoca_core::Expression::FunctionCall {
            name, args, span, ..
        } = expr
        else {
            return None;
        };
        let [
            rumoca_core::Expression::VarRef {
                name: arg_name,
                subscripts,
                ..
            },
        ] = args.as_slice()
        else {
            return None;
        };
        let var = rumoca_core::VarName::new(arg_name.as_str());
        match name.as_str() {
            "inStream" => Some(self.connections.in_stream(&var, subscripts, *span, 0)),
            "actualStream" => Some(self.connections.actual_stream(&var, subscripts, *span)),
            _ => None,
        }
    }
}

impl rumoca_core::ExpressionRewriter for StreamOperatorRewriter<'_> {
    fn rewrite_expression(&mut self, expr: &rumoca_core::Expression) -> rumoca_core::Expression {
        match self.resolve(expr) {
            Some(Ok(resolved)) => resolved,
            Some(Err(error)) => {
                self.error.get_or_insert(error);
                expr.clone()
            }
            None => self.walk_expression(expr),
        }
    }
}

fn member_value(
    var: &rumoca_core::VarName,
    subscripts: &[rumoca_core::Subscript],
    span: Span,
) -> rumoca_core::Expression {
    rumoca_core::Expression::VarRef {
        name: var.clone().into(),
        subscripts: subscripts.to_vec(),
        span,
    }
}

fn real_literal(value: f64, span: Span) -> rumoca_core::Expression {
    rumoca_core::Expression::Literal {
        value: rumoca_core::Literal::Real(value),
        span,
    }
}

fn binary(
    op: rumoca_core::OpBinary,
    lhs: rumoca_core::Expression,
    rhs: rumoca_core::Expression,
    span: Span,
) -> rumoca_core::Expression {
    rumoca_core::Expression::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
        span,
    }
}

fn sum(terms: Vec<rumoca_core::Expression>, span: Span) -> rumoca_core::Expression {
    terms
        .into_iter()
        .reduce(|total, term| binary(rumoca_core::OpBinary::Add, total, term, span))
        .unwrap_or_else(|| real_literal(0.0, span))
}
