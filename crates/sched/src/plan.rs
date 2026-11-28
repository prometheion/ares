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
    /// The most rows of a step that holds the rows of more than one
    /// sequence. A model whose step wider than its own memory borrows memory
    /// for its time (Flash-Next's expert slots lend VRAM, and a prompt is
    /// read layer by layer there) runs such a step for one prompt alone; the
    /// model's own rows ([`Limits::max_rows`] for any other model).
    pub shared_rows: u32,
    /// The most prompt rows in a step that also carries generating
    /// sequences: a piece of prompt (see [`Share`](crate::Share)).
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
}

/// The rows one sequence gets in a step. They are consecutive positions of
/// that sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub seq: SeqId,
    pub rows: u32,
    pub kind: PartKind,
}

/// A step: its parts in row order (generating sequences, then prompts). Sequences that asked and are not in the plan wait for the next
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
/// 2. Drafts, to the generating sequences that asked, in the same order as 1.
/// 3. Prompts, capped while anyone is generating, shortest first so short
///    requests are not stuck behind a long document, except that a prompt
///    that has starved goes first. The forks of a decision are prompts too
///    (short ones, which go early).
///
/// The parts of the plan are in row order, which differs from the order of
/// the grants because the drafts of a sequence follow its newest token:
/// generating sequences (with their drafts), then prompts.
///
/// A step wider than [`Limits::shared_rows`] holds one sequence's rows: when
/// the grants above give more than one sequence rows beyond it, the step is
/// planned again within it.
pub fn plan(requests: &[Request], limits: &Limits) -> Plan {
    let plan = plan_within(requests, limits, limits.max_rows);
    if plan.parts.len() > 1 && plan.rows() > limits.shared_rows {
        return plan_within(requests, limits, limits.shared_rows.min(limits.max_rows));
    }
    plan
}

/// [`plan`] for a step of at most `max_rows` rows.
fn plan_within(requests: &[Request], limits: &Limits, max_rows: u32) -> Plan {
    let mut parts = Vec::new();
    let mut left = max_rows;

    // Generating sequences, longest wait first; the order is stable so equal
    // waits keep the caller's order.
    let mut windows: Vec<(usize, &Request, u32)> = requests
        .iter()
        .filter_map(|r| match r.want {
            Want::Window { drafts } => Some((r, drafts)),
            Want::Prefill { .. } => None,
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

    const LIMITS: Limits = Limits {
        max_rows: 64,
        shared_rows: 64,
        prefill_rows_with_windows: 24,
        starvation_steps: 8,
    };

    fn window(seq: u32, drafts: u32) -> Request {
        Request { seq: SeqId(seq), want: Want::Window { drafts }, waited_steps: 0, priority: 0 }
    }

    fn prefill(seq: u32, tokens: u32, waited_steps: u32) -> Request {
        Request { seq: SeqId(seq), want: Want::Prefill { tokens }, waited_steps, priority: 0 }
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
        let requests = [waited, urgent(window(2, 4)), window(3, 0)];
        let plan = super::plan(&requests, &Limits { max_rows: 2, ..limits });
        assert_eq!(plan.parts, [part(2, 1, PartKind::Window), part(1, 1, PartKind::Window)]);
        let limits = Limits { max_rows: 4, ..LIMITS };
        let plan = super::plan(&[window(1, 4), urgent(window(2, 4))], &limits);
        assert_eq!(plan.parts, [part(2, 3, PartKind::Window), part(1, 1, PartKind::Window)]);
    }

    #[test]
    fn a_step_wider_than_the_shared_rows_holds_one_sequence() {
        // A model that borrows memory for a step wider than 16 rows: a prompt
        // alone takes the widest step; two prompts, or a prompt beside a
        // generating sequence, share a step of 16 rows at most, the shortest
        // prompt first.
        let limits = Limits { shared_rows: 16, ..LIMITS };
        let alone = plan(&[prefill(1, 500, 0)], &limits);
        assert_eq!(alone.parts, [part(1, 64, PartKind::Prefill { completes: false })]);
        let two = plan(&[prefill(1, 500, 0), prefill(2, 10, 0)], &limits);
        assert_eq!(
            two.parts,
            [
                part(2, 10, PartKind::Prefill { completes: true }),
                part(1, 6, PartKind::Prefill { completes: false }),
            ]
        );
        assert!(two.rows() <= 16);
        let short = plan(&[prefill(1, 5, 0), prefill(2, 10, 0)], &limits);
        assert_eq!(short.rows(), 15, "steps within the shared rows are as before");
        let mixed = plan(&[prefill(5, 40, 0), window(1, 0)], &limits);
        assert_eq!(
            mixed.parts,
            [part(1, 1, PartKind::Window), part(5, 15, PartKind::Prefill { completes: false })]
        );
    }

    #[test]
    fn nothing_to_do_is_an_empty_plan() {
        assert_eq!(plan(&[], &LIMITS), Plan::default());
        assert_eq!(plan(&[prefill(1, 0, 0)], &LIMITS).rows(), 0);
    }
}
