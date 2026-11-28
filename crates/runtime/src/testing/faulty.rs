//! A model that fails on purpose, to see that a failure of a device thread
//! reaches the caller as an error that names the device and the cause, and
//! that nothing waits for ever for a thread that is gone.

use ares_api::{EngineError, ModelCard, Prompt};
use ares_state::PageLayout;

use crate::model::{
    ContextSpec, DeviceContext, DeviceModel, LinkSpec, Logits, Model, ModelDrafts, ModelError,
    PrefixCache, Split, StepContext,
};

/// How the part fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The part is built and then the call to `open` fails (an allocation
    /// after the model's own, say).
    Open,
    /// The part's `enqueue` returns an error, before it has enqueued
    /// anything of the step: the other devices of a split have enqueued
    /// their halves, and their kernels wait for this one's.
    Error,
    /// The part's `enqueue` panics, where panics unwind (not in the release
    /// profile, which aborts).
    Panic,
}

/// What fails, where and when.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    /// The device part that fails, as the model numbers them.
    pub device: usize,
    /// The step of that part that fails, counted from 1 by the calls of
    /// `enqueue` it has had (the runtime's own warm-up steps are the first).
    /// Ignored for [`Failure::Open`].
    pub step: u64,
    pub how: Failure,
}

/// `M`, with one device part that fails as `fault` says and otherwise does
/// what `M`'s does.
#[derive(Debug)]
pub struct Faulty<M> {
    inner: M,
    fault: Fault,
}

impl<M: Model> Faulty<M> {
    pub fn new(inner: M, fault: Fault) -> Self {
        Self { inner, fault }
    }
}

impl<M: Model> Model for Faulty<M> {
    fn card(&self) -> ModelCard {
        self.inner.card()
    }

    fn vocab(&self) -> u32 {
        self.inner.vocab()
    }

    fn ends(&self, token: u32) -> bool {
        self.inner.ends(token)
    }

    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        self.inner.encode(prompt)
    }

    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        self.inner.piece(token, text);
    }

    fn context(&self) -> ContextSpec {
        self.inner.context()
    }

    fn split(&self) -> Split {
        self.inner.split()
    }

    fn devices(&self) -> usize {
        self.inner.devices()
    }

    fn links(&self) -> LinkSpec {
        self.inner.links()
    }

    fn max_rows(&self) -> u32 {
        self.inner.max_rows()
    }

    fn logit_rows(&self) -> u32 {
        self.inner.logit_rows()
    }

    fn draft_rows(&self) -> u32 {
        self.inner.draft_rows()
    }

    fn sampling(&self) -> ares_sampler::SampleParams {
        self.inner.sampling()
    }

    fn attention_span(&self) -> u32 {
        self.inner.attention_span()
    }

    fn drafters(&self) -> Vec<crate::Drafter> {
        self.inner.drafters()
    }

    fn prefix_cache(&self) -> Option<PrefixCache> {
        self.inner.prefix_cache()
    }

    fn page_entry(&self, page: u32) -> u32 {
        self.inner.page_entry(page)
    }

    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        self.inner.page_layout(cells_per_page)
    }

    fn abandon(&self, part: usize) {
        self.inner.abandon(part);
    }

    fn measured(&self) {
        self.inner.measured();
    }

    fn idle(&self) {
        self.inner.idle();
    }

    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        let fails = cx.index == self.fault.device;
        let inner = self.inner.open(cx)?;
        if fails && self.fault.how == Failure::Open {
            // The part is built, and then something the runtime does after
            // it fails: its buffers are gone with this error.
            return Err(ModelError::Invalid("injected failure while opening the part".into()));
        }
        Ok(Box::new(FaultyPart { inner, fault: fails.then_some(self.fault), steps: 0 }))
    }
}

struct FaultyPart {
    inner: Box<dyn DeviceModel>,
    fault: Option<Fault>,
    steps: u64,
}

impl DeviceModel for FaultyPart {
    fn logits_per_row(&self) -> u32 {
        self.inner.logits_per_row()
    }

    fn enqueue(&mut self, step: &StepContext<'_>) -> Result<Option<Logits>, ModelError> {
        self.steps += 1;
        if let Some(fault) = self.fault.filter(|f| f.step == self.steps) {
            match fault.how {
                Failure::Error => {
                    return Err(ModelError::Invalid(format!(
                        "injected failure in step {}",
                        self.steps
                    )));
                }
                Failure::Panic => panic!("injected panic in step {}", self.steps),
                Failure::Open => {}
            }
        }
        self.inner.enqueue(step)
    }

    fn logit_tokens(&self) -> ares_cuda::DevicePtr {
        self.inner.logit_tokens()
    }

    fn model_drafts(&self) -> Option<ModelDrafts> {
        self.inner.model_drafts()
    }

    fn model_draft_ms(&mut self) -> f64 {
        self.inner.model_draft_ms()
    }

    fn step_failed(&mut self) {
        self.inner.step_failed();
    }
}
