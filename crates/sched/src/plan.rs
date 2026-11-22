//! Composing a step.

/// Identifies a sequence to the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(pub u32);

/// What a sequence wants from the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// It has `tokens` prompt tokens that are not computed yet.
    Prefill { tokens: u32 },
    /// It is generating: it needs one row for its newest token and would like
    /// `drafts` more rows to verify proposed tokens in the same step.
    Window { drafts: u32 },
    /// A decision whose prompt is computed. It needs `rows` more rows scored:
    /// the rows of the forks of its options, which read the prompt's pages
    /// and whose logits give each option its score. Every row needs its
    /// logits and none is optional: without all of them the decision has no
    /// answer. `seq` is the decision's, not a fork's; the runtime lays out
    /// the forks.
    Score { rows: u32 },
}

/// One sequence's request for the next step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub seq: SeqId,
    pub want: Want,
    /// How many steps in a row this sequence got nothing.
    pub waited_steps: u32,
    /// Who goes first, a lower number before a higher one: every order below
    /// is by priority first.
    pub priority: i32,
}

/// The bounds a step is planned within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The most rows a step can hold (the size of the step's scratch memory).
    pub max_rows: u32,
    /// The most prompt rows in a step that also carries generating
    /// sequences or decisions: a piece of prompt (see [`Share`](crate::Share)).
    pub prefill_rows_with_windows: u32,
    /// A prompt that waited this many steps goes before shorter prompts.
    pub starvation_steps: u32,
}

/// What part of a step is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartKind {
    /// Prompt rows. `completes` is set when these are the last rows of the
    /// prompt, so the step must produce the logits of the last row.
    Prefill { completes: bool },
    /// The newest token and `rows - 1` drafts of a generating sequence. Every
    /// row's logits are needed.
    Window,
    /// Option rows of a decision. Every row's logits are needed. `completes`
    /// is set when these are the last rows the decision asked for, so its
    /// answer can be made from this step; otherwise the rest is asked for
    /// again ([`Want::Score`] with the rows that are left) in a later step.
    Score { completes: bool },
}

/// The rows one sequence gets in a step. They are consecutive positions of
/// that sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub seq: SeqId,
    pub rows: u32,
    pub kind: PartKind,
}

/// A step: its parts in row order (generating sequences, then decisions, then
/// prompts). Sequences that asked and are not in the plan wait for the next
/// step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    pub parts: Vec<Part>,
}

impl Plan {
    pub fn rows(&self) -> u32 {
        self.parts.iter().map(|p| p.rows).sum()
    }
}

/// Plans the next step.
///
/// Rows are granted in this order, and within each by priority first (a
/// lower number first; see [`Request::priority`]):
///
/// 1. The newest token of every generating sequence: a person is reading
///    their output, and a row of theirs is cheap.
/// 2. The rows of decisions ([`Want::Score`]), the longest wait first. A
///    decision is answered once, when all its rows have run, so its rows are
///    not like drafts that merely speed something up: they come before drafts.
///    A decision whose rows do not all fit gets as many as fit, and asks for
///    the rest in the next step.
/// 3. Drafts, to the generating sequences that asked, in the same order as 1.
/// 4. Prompts, capped while anyone is generating or deciding, shortest first
///    so short requests are not stuck behind a long document, except that a
///    prompt that has starved goes first.
///
/// The parts of the plan are in row order, which differs from the order of
/// the grants because the drafts of a sequence follow its newest token:
/// generating sequences (with their drafts), then decisions, then prompts.
pub fn plan(requests: &[Request], limits: &Limits) -> Plan {
    let mut parts = Vec::new();
    let mut left = limits.max_rows;

    // Generating sequences, longest wait first; the order is stable so equal
    // waits keep the caller's order.
    let mut windows: Vec<(usize, &Request, u32)> = requests
        .iter()
        .filter_map(|r| match r.want {
            Want::Window { drafts } => Some((r, drafts)),
            Want::Prefill { .. } | Want::Score { .. } => None,
        })
        .enumerate()
        .map(|(i, (r, drafts))| (i, r, drafts))
        .collect();
    windows.sort_by_key(|(i, r, _)| (r.priority, std::cmp::Reverse(r.waited_steps), *i));

    let first_window = parts.len();
    for (_, request, _) in &windows {
        if left == 0 {
            break;
        }
        parts.push(Part { seq: request.seq, rows: 1, kind: PartKind::Window });
        left -= 1;
    }
    let after_windows = parts.len();

    // Decisions, before any draft is granted. `sort_by_key` is stable, so
    // equal waits keep the caller's order.
    let mut decisions: Vec<(&Request, u32)> = requests
        .iter()
        .filter_map(|r| match r.want {
            Want::Score { rows } if rows > 0 => Some((r, rows)),
            _ => None,
        })
        .collect();
    decisions.sort_by_key(|(r, _)| (r.priority, std::cmp::Reverse(r.waited_steps)));
    for (request, rows) in decisions {
        if left == 0 {
            break;
        }
        let granted = rows.min(left);
        left -= granted;
        parts.push(Part {
            seq: request.seq,
            rows: granted,
            kind: PartKind::Score { completes: granted == rows },
        });
    }

    for (part, (_, _, drafts)) in parts[first_window..after_windows].iter_mut().zip(&windows) {
        let extra = (*drafts).min(left);
        part.rows += extra;
        left -= extra;
    }
    let has_generation = !parts.is_empty();

    // Prompts.
    let mut budget = if has_generation { left.min(limits.prefill_rows_with_windows) } else { left };
    let mut prompts: Vec<(&Request, u32)> = requests
        .iter()
        .filter_map(|r| match r.want {
            Want::Prefill { tokens } if tokens > 0 => Some((r, tokens)),
            _ => None,
        })
        .collect();
    prompts.sort_by_key(|(r, tokens)| {
        let starved = r.waited_steps >= limits.starvation_steps;
        // By priority; then starved prompts first, the longest wait among
        // them first; then by remaining length.
        (r.priority, !starved, if starved { u32::MAX - r.waited_steps } else { *tokens })
    });
    for (request, tokens) in prompts {
        if budget == 0 {
            break;
        }
        let rows = tokens.min(budget);
        budget -= rows;
        parts.push(Part {
            seq: request.seq,
            rows,
            kind: PartKind::Prefill { completes: rows == tokens },
        });
    }
    Plan { parts }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Limits =
        Limits { max_rows: 64, prefill_rows_with_windows: 24, starvation_steps: 8 };

    fn window(seq: u32, drafts: u32) -> Request {
        Request { seq: SeqId(seq), want: Want::Window { drafts }, waited_steps: 0, priority: 0 }
    }

    fn prefill(seq: u32, tokens: u32, waited_steps: u32) -> Request {
        Request { seq: SeqId(seq), want: Want::Prefill { tokens }, waited_steps, priority: 0 }
    }

    fn score(seq: u32, rows: u32, waited_steps: u32) -> Request {
        Request { seq: SeqId(seq), want: Want::Score { rows }, waited_steps, priority: 0 }
    }

    fn part(seq: u32, rows: u32, kind: PartKind) -> Part {
        Part { seq: SeqId(seq), rows, kind }
    }

    #[test]
    fn a_prompt_alone_gets_the_whole_step() {
        let plan = plan(&[prefill(1, 1000, 0)], &LIMITS);
        assert_eq!(plan.parts, [part(1, 64, PartKind::Prefill { completes: false })]);
        let last = super::plan(&[prefill(1, 40, 0)], &LIMITS);
        assert_eq!(last.parts, [part(1, 40, PartKind::Prefill { completes: true })]);
    }

    #[test]
    fn windows_of_unequal_length_share_a_step_with_a_bounded_slice_of_a_prompt() {
        let requests = [prefill(9, 5000, 0), window(1, 2), window(2, 0), window(3, 5)];
        let plan = plan(&requests, &LIMITS);
        assert_eq!(
            plan.parts,
            [
                part(1, 3, PartKind::Window),
                part(2, 1, PartKind::Window),
                part(3, 6, PartKind::Window),
                part(9, 24, PartKind::Prefill { completes: false }),
            ]
        );
        assert_eq!(plan.rows(), 34);
    }

    #[test]
    fn every_generating_sequence_gets_its_token_before_anyone_gets_drafts() {
        let limits = Limits { max_rows: 5, ..LIMITS };
        let requests = [window(1, 4), window(2, 4), window(3, 4), prefill(9, 100, 0)];
        let plan = plan(&requests, &limits);
        assert_eq!(
            plan.parts,
            [
                part(1, 3, PartKind::Window),
                part(2, 1, PartKind::Window),
                part(3, 1, PartKind::Window)
            ]
        );

        // More generating sequences than rows: the ones that waited go first.
        let limits = Limits { max_rows: 2, ..LIMITS };
        let mut requests = [window(1, 0), window(2, 0), window(3, 0)];
        requests[2].waited_steps = 1;
        let plan = super::plan(&requests, &limits);
        assert_eq!(plan.parts, [part(3, 1, PartKind::Window), part(1, 1, PartKind::Window)]);
    }

    #[test]
    fn short_prompts_go_first_unless_a_long_one_has_starved() {
        let requests = [prefill(1, 500, 0), prefill(2, 10, 0), prefill(3, 30, 0)];
        let plan = plan(&requests, &LIMITS);
        assert_eq!(
            plan.parts,
            [
                part(2, 10, PartKind::Prefill { completes: true }),
                part(3, 30, PartKind::Prefill { completes: true }),
                part(1, 24, PartKind::Prefill { completes: false }),
            ]
        );

        let requests = [prefill(1, 500, 8), prefill(2, 10, 0)];
        let plan = super::plan(&requests, &LIMITS);
        assert_eq!(plan.parts, [part(1, 64, PartKind::Prefill { completes: false })]);
    }

    #[test]
    fn a_step_can_mix_drafts_a_decision_and_a_prompt() {
        // Two generating sequences (one with drafts), a decision of 6 option
        // rows and a long prompt: every kind of demand in one step.
        let requests = [prefill(9, 5000, 0), window(1, 2), score(5, 6, 0), window(2, 0)];
        let plan = plan(&requests, &LIMITS);
        assert_eq!(
            plan.parts,
            [
                part(1, 3, PartKind::Window),
                part(2, 1, PartKind::Window),
                part(5, 6, PartKind::Score { completes: true }),
                part(9, 24, PartKind::Prefill { completes: false }),
            ]
        );
        assert_eq!(plan.rows(), 34);
    }

    #[test]
    fn decision_rows_come_before_drafts_but_after_every_newest_token() {
        // 6 rows: the two newest tokens first, then the decision's 3 rows,
        // and only the single row that is left goes to drafts.
        let limits = Limits { max_rows: 6, ..LIMITS };
        let requests = [window(1, 4), score(5, 3, 0), window(2, 4)];
        let plan = plan(&requests, &limits);
        assert_eq!(
            plan.parts,
            [
                part(1, 2, PartKind::Window),
                part(2, 1, PartKind::Window),
                part(5, 3, PartKind::Score { completes: true }),
            ]
        );
        assert_eq!(plan.rows(), 6);

        // Not even one decision row is granted before everyone's newest token.
        let limits = Limits { max_rows: 2, ..LIMITS };
        let plan = super::plan(&[score(5, 3, 9), window(1, 0), window(2, 0)], &limits);
        assert_eq!(plan.parts, [part(1, 1, PartKind::Window), part(2, 1, PartKind::Window)]);
    }

    #[test]
    fn a_decision_that_does_not_fit_is_split_over_two_steps() {
        let limits = Limits { max_rows: 10, ..LIMITS };
        // Step 1: one generating sequence takes 1 row, so 9 of the 12 option
        // rows fit. The decision is not complete.
        let first = plan(&[window(1, 0), score(7, 12, 0)], &limits);
        assert_eq!(
            first.parts,
            [part(1, 1, PartKind::Window), part(7, 9, PartKind::Score { completes: false })]
        );

        // Step 2: the caller asks for the 3 rows that are left. They fit
        // with room to spare, and the decision completes.
        let second = super::plan(&[window(1, 0), score(7, 3, 0)], &limits);
        assert_eq!(
            second.parts,
            [part(1, 1, PartKind::Window), part(7, 3, PartKind::Score { completes: true })]
        );
    }

    #[test]
    fn the_decision_that_waited_longest_gets_rows_first() {
        let limits = Limits { max_rows: 8, ..LIMITS };
        let requests = [score(1, 5, 0), score(2, 5, 3), score(3, 5, 1)];
        let plan = plan(&requests, &limits);
        assert_eq!(
            plan.parts,
            [
                part(2, 5, PartKind::Score { completes: true }),
                part(3, 3, PartKind::Score { completes: false }),
            ],
            "the one that waited longest is whole, the next takes what is left, the last waits"
        );

        // Equal waits keep the caller's order.
        let plan = super::plan(&[score(4, 5, 2), score(3, 5, 2)], &limits);
        assert_eq!(plan.parts[0].seq, SeqId(4));
    }

    #[test]
    fn decisions_cap_the_prompt_share_like_generation_does() {
        // No generating sequence at all: a decision alone still limits the
        // prompt to the share, where a prompt alone would get the whole step.
        let plan = plan(&[score(5, 4, 0), prefill(9, 5000, 0)], &LIMITS);
        assert_eq!(
            plan.parts,
            [
                part(5, 4, PartKind::Score { completes: true }),
                part(9, 24, PartKind::Prefill { completes: false }),
            ]
        );
        let alone = super::plan(&[prefill(9, 5000, 0)], &LIMITS);
        assert_eq!(alone.parts, [part(9, 64, PartKind::Prefill { completes: false })]);
    }

    #[test]
    fn a_decision_asking_for_no_rows_is_not_in_the_plan() {
        let plan = plan(&[score(5, 0, 0), prefill(9, 10, 0)], &LIMITS);
        assert_eq!(plan.parts, [part(9, 10, PartKind::Prefill { completes: true })]);
    }

    #[test]
    fn a_better_priority_goes_first_in_every_kind_of_row() {
        let urgent = |mut r: Request| {
            r.priority = -1;
            r
        };
        // A long prompt of a better priority goes before a short one, and
        // before a starved one of the default priority.
        let requests = [prefill(1, 10, 0), urgent(prefill(2, 500, 0)), prefill(3, 30, 9)];
        let plan = plan(&requests, &LIMITS);
        assert_eq!(plan.parts, [part(2, 64, PartKind::Prefill { completes: false })]);

        // Too few rows for every newest token: the better priority's first,
        // whoever waited longer; its drafts too.
        let limits = Limits { max_rows: 3, ..LIMITS };
        let mut waited = window(1, 4);
        waited.waited_steps = 5;
        let requests = [waited, urgent(window(2, 4)), urgent(score(3, 1, 0))];
        let plan = super::plan(&requests, &limits);
        assert_eq!(
            plan.parts,
            [
                part(2, 1, PartKind::Window),
                part(1, 1, PartKind::Window),
                part(3, 1, PartKind::Score { completes: true }),
            ]
        );
        let limits = Limits { max_rows: 4, ..LIMITS };
        let plan = super::plan(&[window(1, 4), urgent(window(2, 4))], &limits);
        assert_eq!(plan.parts, [part(2, 3, PartKind::Window), part(1, 1, PartKind::Window)]);
    }

    #[test]
    fn nothing_to_do_is_an_empty_plan() {
        assert_eq!(plan(&[], &LIMITS), Plan::default());
        assert_eq!(plan(&[prefill(1, 0, 0)], &LIMITS).rows(), 0);
    }
}
