//! Steps wider than the decode kernels take: slices of a prompt.
//!
//! A step of up to [`MAX_BATCH`] rows runs as the module above describes. A wider
//! one runs the same layers with the kernels that take any number of rows:
//!
//! - **Dense products** go through the tile kernels (`Kernels::product`) in the form of
//!   activations the device and the matrix type ask for ([`ActivationForm`]): the int8
//!   rows the fused kernels already write, or f16 rows converted from their f32
//!   section. The float matrices (the mixers', the router, the projections of the indexer
//!   and of the n-gram block, the recurrent gates) take the float tile kernels: their input
//!   is converted to f16 for the product (they read the f32 values a fused kernel wrote,
//!   whatever form the quantized matrices take) and the sums are f32. A matrix with a weight
//!   that f16 cannot hold ([`Mat::range`], read when the model was loaded), or a width the
//!   tile kernels do not take, is multiplied by the decode kernel eight rows at a time, the
//!   chunks spread over four streams. A wide step reads
//!   its stream mixers with the separate kernels and not with the fused read (the fused one
//!   is made for a few rows). A mixer that shares the layer pushes the mixed input to its
//!   worker as f32 rows with a flag per row, and the worker prepares it
//!   ([`Part::prepare_arrival`]).
//! - **The router** is such a float product and picks for every row
//!   ([`ExpertKernels::pick_rows`]): it leaves the ids and weights of the picks, not the
//!   groups of the decode kernels, which hold eight rows.
//! - **The routed experts** run on the many-row kernels ([`ExpertRowKernels`]) in rounds.
//!   Those kernels do not wait on the device for an expert that has no slot, so the host
//!   takes part in every layer: the first round computes the experts that are in a slot
//!   when the layer is enqueued (the host gives their addresses, [`Part::experts_round`]);
//!   the host reads the picks (they reach it as the record of the decode kernels does),
//!   brings the experts that were missing and gives a table of just those for the next
//!   round, which adds to the same output. The plan is made once per layer.

use ares_cuda::{ExpertShape, PrefillPath};

use super::*;

/// Address tables alive at once: the round being computed and the one prepared next.
pub(super) const ROUNDS: u64 = 2;

/// The most rows of a wide step in which the CPU may compute misses, by default
/// ([`Options::cpu_rows`]). The CPU takes experts that few rows picked; in a step of more
/// rows than this few experts are, and the buffers grow with the rows.
pub(crate) const CPU_ROWS: u32 = 2048;

/// Page-locked bytes for the CPU's part of wide steps of up to `rows` rows of `n` values:
/// the rows' activations as the CPU reads them (int8 codes and their scales) and their sums.
pub(super) fn cpu_host_bytes(n: u64, rows: u32) -> u64 {
    let act = ares_runtime::expert_store::cpu::act_bytes(n as usize, rows as usize) as u64;
    act.next_multiple_of(64) + u64::from(rows) * n * 4
}

/// Bytes of the record of a wide step: a header, then the ids and the weights of the picks.
/// It is what the router leaves on the device and what reaches the host.
pub(super) fn record_bytes(shape: &ExpertShape, rows: u32) -> u64 {
    16 + 8 * u64::from(shape.top_k) * u64::from(rows)
}

/// Bytes of the plan of the widest step: the smallest tile has the most items.
pub(super) fn plan_bytes(shape: &ExpertShape, rows: u32) -> u64 {
    shape.rows_plan_bytes(rows, ares_cuda::ROWS_TILES[0])
}

/// Bytes of the scratch that holds the f16 form of the input of a float product
/// ([`Part::wide_product`]) for steps of up to `max_rows` rows: of the widest input a float
/// matrix reads, which is the normalized streams (`wide` values) on a part that has them and
/// the mixed input (`n` values) on a worker.
pub(super) fn float_f16_bytes(streams: bool, wide: u64, n: u64, max_rows: u32) -> u64 {
    Kernels::prefill_activation_bytes(PrefillPath::F16, float_width(streams, wide, n), max_rows)
}

/// The widest input the scratch of [`float_f16_bytes`] holds.
fn float_width(streams: bool, wide: u64, n: u64) -> u32 {
    (if streams { wide.max(n) } else { n }) as u32
}

/// A layer of a wide step whose experts the host is bringing.
#[derive(Clone, Copy, Debug)]
struct Layer {
    index: usize,
    tile: u32,
    rounds: u32,
}

/// What a part needs for wide steps.
pub(super) struct State {
    kernels: ExpertRowKernels,
    precision: Precision,
    form: ActivationForm,
    /// The widest input of a float product the scratch for its f16 form holds.
    float_width: u32,
    /// The address tables of a layer as the host wrote them: per layer
    /// [`ROUNDS`] of them, in page-locked memory the copies read.
    staging: PinnedBuffer,
    /// Recorded behind the latest round of experts.
    done: Event,
    layer: Option<Layer>,
    /// Streams that run the chunks of eight rows of a float product side by side: such a
    /// kernel has a few dozen blocks, so one after the other they leave the device idle.
    helpers: Vec<Stream>,
    /// Recorded on the step's stream before the helpers start, and on each helper behind
    /// its chunks.
    fork: Event,
    joins: Vec<Event>,
    /// The CPU's part of wide steps of up to `cpu_rows` rows (0: none): the mixed input of
    /// a layer's rows as the CPU reads it, copied when the layer is enqueued, and the sums
    /// the host gives back, in page-locked memory; the events behind the copies.
    cpu_rows: u32,
    host_act: PinnedBuffer,
    host_sums: PinnedBuffer,
    act_ready: Event,
    act_copied: Event,
    sums_copied: Event,
    /// A copy of the mixed input was enqueued for the layer being computed, and sums were
    /// copied since the host last wrote them.
    act_pending: bool,
    sums_pending: bool,
}

/// Streams the chunks of a float product are spread over.
const HELPERS: usize = 4;

/// The ggml types of the quantized dense matrices that read prepared rows in wide steps: one
/// form of the rows serves all of them, so every one of them has a say in it (the recurrent
/// gates and the indexer's projections included, which most files store as floats).
fn dense_types(w: &DeviceWeights) -> Vec<GgmlType> {
    let mut types = Vec::new();
    for layer in &w.layers {
        match &layer.core {
            CoreW::Recurrent(r) => {
                types.extend([r.qkv.ty, r.z.ty, r.alpha.ty, r.beta.ty, r.out.ty])
            }
            CoreW::Attention(a) => {
                types.extend([a.q.ty, a.k.ty, a.v.ty, a.out.ty, a.indexer_q.ty, a.indexer_k.ty])
            }
            CoreW::None => {}
        }
        if let Some(shared) = &layer.moe.shared {
            types.extend([shared.gate.ty, shared.up.ty, shared.down.ty]);
        }
    }
    types.retain(|ty| !is_float(*ty));
    types.sort_by_key(|ty| ty.name());
    types.dedup();
    types
}

fn is_float(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

impl State {
    /// The state for wide steps of a part, or an error naming what the device lacks.
    pub(super) fn open(
        device: &Device,
        kernels: &Kernels,
        w: &DeviceWeights,
        opts: &Options,
        frame: &Frame,
    ) -> Result<Self, ModelError> {
        let rows = ExpertRowKernels::load(device).map_err(|e| {
            invalid(format!("steps of {} rows need the many-row kernels: {e}", opts.max_rows))
        })?;
        if !kernels.has_prefill_kernels() {
            return Err(invalid(format!(
                "steps of {} rows need the tile kernels, which {} lacks",
                opts.max_rows,
                device.info().name
            )));
        }
        for layer in &w.layers {
            layer.moe.shape.check_rows().map_err(|e| {
                invalid(format!("steps of {} rows, layer {}: {e}", opts.max_rows, layer.layer))
            })?;
        }
        let types = dense_types(w);
        // The path of the tile kernels each quantized dense type takes (the float
        // matrices take the f16 path of `Part::wide_float`), measured on the device. The
        // weights are on the device already: where the probe finds no room for its
        // matrix, the type keeps the rule of a type that was not measured.
        if opts.precision == Precision::Fast {
            for ty in &types {
                if !is_float(*ty) && kernels.has_prefill_path(*ty, PrefillPath::F16) {
                    match kernels.measure_prefill_path(*ty) {
                        Err(e) if e.is_out_of_memory() => {}
                        other => {
                            other?;
                        }
                    }
                }
            }
        }
        // One form of activations for every dense matrix, so that each fused kernel writes
        // one: the form the options' precision gives all of them, else the exact one, which
        // is f16 for all.
        let form_of = |precision: Precision, ty: GgmlType| {
            kernels
                .activation_form(ty, WeightLayout::File, precision, opts.max_rows)
                .map_err(|e| invalid(e.to_string()))
        };
        let forms =
            types.iter().map(|ty| form_of(opts.precision, *ty)).collect::<Result<Vec<_>, _>>()?;
        let (precision, form) = match forms.first() {
            Some(first) if forms.iter().all(|f| f == first) => (opts.precision, *first),
            Some(_) => (Precision::Exact, ActivationForm::F16),
            None => (opts.precision, ActivationForm::Int8),
        };
        let layers = w.layers.len();
        // The widths the frame sized the scratch of the float products with
        // ([`float_f16_bytes`]): the streams' where the part has the mixers.
        let n = w.layers.first().map_or(0, |l| l.moe.shape.n_embd);
        let streams =
            w.layers.iter().filter_map(|l| l.mixers.map(|(attn, _)| attn.down.n_in)).max();
        let float_width =
            float_width(streams.is_some(), u64::from(streams.unwrap_or(0)), u64::from(n));
        let cpu_rows = opts.cpu_rows.min(opts.max_rows);
        let act_bytes = ares_runtime::expert_store::cpu::act_bytes(n as usize, cpu_rows as usize);
        Ok(Self {
            cpu_rows,
            host_act: PinnedBuffer::new(device, act_bytes.max(16))?,
            host_sums: PinnedBuffer::new(device, (cpu_rows as usize * n as usize * 4).max(16))?,
            act_ready: Event::new(device)?,
            act_copied: Event::new(device)?,
            sums_copied: Event::new(device)?,
            act_pending: false,
            sums_pending: false,
            kernels: rows,
            precision,
            form,
            float_width,
            staging: PinnedBuffer::new(device, (frame.addr_stage as usize * layers).max(16))?,
            done: Event::new(device)?,
            layer: None,
            helpers: (0..HELPERS).map(|_| Stream::new(device, 0)).collect::<Result<_, _>>()?,
            fork: Event::new(device)?,
            joins: (0..HELPERS).map(|_| Event::new(device)).collect::<Result<_, _>>()?,
        })
    }
}

impl Part {
    /// Whether the dense products of a step of `rows` rows run on the tile kernels.
    pub(super) fn is_wide(rows: u32) -> bool {
        rows > MAX_BATCH
    }

    /// Whether the outputs of the recurrent and attention blocks and of the shared expert
    /// are quantized for their products in f16 instead of int8 (in wide steps).
    pub(super) fn wide_f16(&self, rows: u32) -> bool {
        Self::is_wide(rows)
            && self.wide_state.as_ref().is_some_and(|w| w.form == ActivationForm::F16)
    }

    /// `y = m x` for the `rows` rows of a wide step. `act` is the rows in the int8 layout
    /// (the layout of `Kernels::quantize_activations`, with the f32 values as its last
    /// section) that a fused kernel wrote: the buffer of the mixed input or the one of the
    /// quantized outputs.
    ///
    /// # Safety
    ///
    /// As [`Part::product`].
    pub(super) unsafe fn wide_product(
        &self,
        stream: &Stream,
        m: &Mat,
        act: DevicePtr,
        y: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        let wide =
            self.wide_state.as_ref().ok_or_else(|| invalid("a wide step without wide state"))?;
        if is_float(m.ty) {
            // SAFETY: the caller's guarantee.
            return unsafe { self.wide_float(stream, m, act, y, rows) };
        }
        let s = self.f.s;
        let input = match wide.form {
            ActivationForm::Int8 => act,
            ActivationForm::F16 if act == self.at(s.m) => self.at(s.mixed_f16),
            ActivationForm::F16 if act == self.at(s.act) => self.at(s.out_f16),
            ActivationForm::F16 => return Err(invalid("a wide product of an unprepared input")),
        };
        // SAFETY: the matrix is in the arena as the file stores it; the rows were prepared
        // in the form the product reads.
        unsafe {
            self.kernels.product(
                stream,
                m.ty,
                WeightLayout::File,
                wide.precision,
                self.w.mat(m),
                input,
                y,
                m.n_in,
                m.n_out,
                rows,
            )?;
        }
        Ok(())
    }

    /// [`Part::wide_product`] for a float matrix (F32, F16, BF16), and for every matrix the
    /// step multiplies with unquantized values whatever its type ([`Part::exact_product`]):
    /// it reads the f32 section of the rows, on the tile kernel of the matrix's type with the
    /// rows converted to f16 into a scratch of this product's own. The sums are f32 ([`PrefillPath::F16`]) whatever the
    /// step's precision is: the path with f16 sums loses a row of small weights among larger
    /// ones (the recurrent gates have such rows), and with it the made-up model's layers
    /// were up to four times as far from the reference (`tests/qwen4exp_device.rs`,
    /// `every_layer_of_wide_steps_matches_the_reference`). A matrix with a weight beyond f16
    /// ([`Mat::range`]), or of a width the tile kernel or the scratch does not take, is
    /// multiplied by the decode kernel that reads unquantized rows
    /// ([`Kernels::matvec_exact`]), which computes with the weights as stored: eight rows at
    /// a time, the chunks spread over the helper streams.
    ///
    /// # Safety
    ///
    /// As [`Part::product`].
    pub(super) unsafe fn wide_float(
        &self,
        stream: &Stream,
        m: &Mat,
        act: DevicePtr,
        y: DevicePtr,
        rows: u32,
    ) -> Result<(), ModelError> {
        let w = self.w.mat(m);
        let wide =
            self.wide_state.as_ref().ok_or_else(|| invalid("a wide step without wide state"))?;
        let row_bytes = 4 * u64::from(m.n_in);
        let f32_at =
            |k: u32| Kernels::quantized_activation_bytes(m.n_in, k) - row_bytes * u64::from(k);
        let tile = m.n_in.is_multiple_of(32) && m.n_in <= wide.float_width;
        if let Some(path) = m.range.path(tile.then_some(PrefillPath::F16)) {
            let f16 = self.at(self.f.s.float_f16);
            // SAFETY: the rows are inside the caller's buffer; the scratch holds the f16 form
            // of `float_width` values a row and only this product reads it; the matrix is
            // 64-byte aligned in the arena.
            unsafe {
                self.kernels.prepare_prefill_activations(
                    stream,
                    PrefillPath::F16,
                    act.offset(f32_at(rows)),
                    f16,
                    m.n_in,
                    rows,
                )?;
                self.kernels.matmul(stream, m.ty, path, w, f16, y, m.n_in, m.n_out, rows)?;
            }
            return Ok(());
        }
        wide.fork.record(stream)?;
        for helper in &wide.helpers {
            helper.wait_event(&wide.fork)?;
        }
        let (mut first, mut next) = (0, 0);
        while first < rows {
            let k = (rows - first).min(MAX_BATCH);
            let chunk = act.offset(f32_at(rows) + row_bytes * u64::from(first) - f32_at(k));
            let at = y.offset(4 * u64::from(first) * u64::from(m.n_out));
            // SAFETY: the chunk's rows are inside the caller's buffers, which nothing on
            // the step's stream touches until the helpers have been waited for.
            unsafe {
                let helper = &wide.helpers[next];
                self.kernels.matvec_exact(helper, m.ty, w, chunk, at, m.n_in, m.n_out, k)?;
            }
            first += k;
            next = (next + 1) % wide.helpers.len();
        }
        for (helper, join) in wide.helpers.iter().zip(&wide.joins) {
            join.record(helper)?;
            stream.wait_event(join)?;
        }
        Ok(())
    }

    /// Prepares the mixed input a mixer pushed (f32 rows) in the int8 layout the products
    /// and the router read; a worker does it after its wait, which is the launch before.
    pub(super) fn prepare_arrival(&self, stream: &Stream, rows: u32) -> Result<(), ModelError> {
        if !Self::is_wide(rows) {
            return Ok(());
        }
        let s = self.f.s;
        // SAFETY: `m_raw` holds the rows once the wait before has run; `m` is scratch of
        // this part.
        unsafe {
            self.kernels.prepare_prefill_activations(
                stream,
                ares_cuda::PrefillPath::Int8,
                self.at(s.m_raw),
                self.at(s.m),
                self.n,
                rows,
            )?;
        }
        Ok(())
    }

    /// Converts the mixed input of a half block to f16 where the products read that.
    ///
    /// # Safety
    ///
    /// The mixed input must be at `m` in scratch.
    pub(super) unsafe fn prepare_mixed(
        &self,
        stream: &Stream,
        rows: u32,
    ) -> Result<(), ModelError> {
        if !self.wide_f16(rows) {
            return Ok(());
        }
        let s = self.f.s;
        // SAFETY: the rows were written by the fused read before.
        unsafe {
            self.kernels.prepare_activations(
                stream,
                ActivationForm::F16,
                self.act_values(s.m, self.n, rows),
                self.at(s.mixed_f16),
                self.n,
                rows,
            )?;
        }
        Ok(())
    }

    /// The first stage of the routed experts of a wide step in layer `i`. A part that routes
    /// runs the router and leaves the picks with the host (the board); every part with the
    /// shared expert computes it. The experts themselves run in [`Part::experts_round`].
    pub(super) fn moe_wide(
        &mut self,
        env: &StepEnv<'_>,
        i: usize,
        w: &MoeW,
    ) -> Result<(), ModelError> {
        let stream = env.stream;
        let rows = env.record.dims().n_rows;
        let s = self.f.s;
        let m = self.at(s.m);
        let tile = ExpertRowKernels::tile_for(&w.shape, rows);
        let wide =
            self.wide_state.as_mut().ok_or_else(|| invalid("a wide step without wide state"))?;
        wide.layer = Some(Layer { index: i, tile, rounds: 0 });
        // SAFETY: scratch and weights of this part, sized for the most rows; the mixed
        // input is the fused read's.
        unsafe {
            self.prepare_mixed(stream, rows)?;
            self.copy_cpu_act(stream, rows)?;
            if self.role == Role::Worker {
                // The mixer routes. The worker's output starts from zeros (its exchange
                // cleared it) and the shared expert's is sent on its own, with the gate.
                if let Some(shared) = &w.shared {
                    self.shared(stream, shared, m, rows)?;
                    self.mark(stream, Phase::Dense)?;
                }
                return self.mark(stream, Phase::Experts);
            }
            let router = w.router.ok_or_else(|| invalid("a layer without its router"))?;
            let record = self.at(s.rec_full);
            let bytes = record_bytes(&w.shape, rows);
            // The router is a float matrix like the others: `logits[b][e]` is its row `e`
            // times row `b`.
            let router = Mat {
                ty: GgmlType::F32,
                n_in: w.shape.n_embd,
                n_out: w.shape.n_experts,
                at: router,
                range: w.router_range,
            };
            self.wide_float(stream, &router, m, self.at(s.logits), rows)?;
            self.experts.pick_rows(stream, &w.shape, self.at(s.logits), record, rows)?;
            self.routed[i].record(stream)?;
            self.side.wait_event(&self.routed[i])?;
            self.side.copy_to_host(self.board.record_dst(i), record, bytes as usize)?;
            self.board.copies_enqueued(i, &self.side)?;
            self.enqueue_wide_prediction(stream, i, m, rows)?;
            self.mark(stream, Phase::Experts)?;
            if let Some(shared) = &w.shared {
                self.product(stream, &shared.inp, m, self.at(s.s), rows)?;
                self.shared(stream, shared, m, rows)?;
                self.q4.gate_scale(
                    stream,
                    self.at(s.ysh),
                    self.at(s.s),
                    self.at(s.out),
                    rows,
                    self.n,
                )?;
            } else {
                stream.fill(
                    self.at(s.out),
                    0,
                    (u64::from(rows) * u64::from(self.n) * 4) as usize,
                )?;
            }
        }
        self.mark(stream, Phase::Dense)
    }

    /// Copies the mixed input of a wide step's rows as the CPU reads it (the int8 codes and
    /// their scales) to page-locked memory, on the side stream behind what wrote it: the
    /// host reads it once the layer's misses are known ([`Part::cpu_act`]). Nothing for a
    /// step of more rows than the CPU takes part in.
    ///
    /// # Safety
    ///
    /// The mixed input of the step's rows must be at `m` in scratch, written by what is
    /// enqueued on `stream` so far.
    unsafe fn copy_cpu_act(&mut self, stream: &Stream, rows: u32) -> Result<(), ModelError> {
        let m = self.at(self.f.s.m);
        let n = self.n as usize;
        let wide = self.wide_state.as_mut().ok_or_else(|| invalid("no wide state"))?;
        wide.act_pending = false;
        if rows > wide.cpu_rows {
            return Ok(());
        }
        let bytes = ares_runtime::expert_store::cpu::act_bytes(n, rows as usize);
        wide.act_ready.record(stream)?;
        self.side.wait_event(&wide.act_ready)?;
        // SAFETY: the buffer holds the codes and scales of `cpu_rows` rows; the host reads
        // it only between `act_copied` and the next layer's copy, which is enqueued after
        // the host is done with this layer.
        unsafe { self.side.copy_to_host(wide.host_act.as_mut_ptr(), m, bytes)? };
        wide.act_copied.record(&self.side)?;
        wide.act_pending = true;
        Ok(())
    }

    /// Whether the CPU may compute experts of the layer of the wide step being computed:
    /// the step has few enough rows, and the rows' mixed input is being copied for it.
    pub fn cpu_takes_part(&self) -> bool {
        self.wide_state.as_ref().is_some_and(|w| w.act_pending)
    }

    /// The mixed input of the `rows` rows of the layer being computed as the CPU reads it
    /// (`ares_runtime::expert_store::cpu::act_bytes`), once its copy has arrived; `None`
    /// while it is on its way.
    pub fn cpu_act(&self, rows: u32) -> Result<Option<&[u8]>, ModelError> {
        let wide = self.wide_state.as_ref().ok_or_else(|| invalid("no wide state"))?;
        if !wide.act_pending || rows > wide.cpu_rows {
            return Err(invalid("no copy of the mixed input for the CPU"));
        }
        if !wide.act_copied.is_done()? {
            return Ok(None);
        }
        let bytes = ares_runtime::expert_store::cpu::act_bytes(self.n as usize, rows as usize);
        // SAFETY: the copy has run, nothing writes the buffer before the next layer is
        // enqueued, and it holds `bytes`.
        Ok(Some(unsafe { std::slice::from_raw_parts(wide.host_act.as_ptr(), bytes) }))
    }

    /// Adds `sums`, what the CPU computed for the rows from `first` on of the wide step (a
    /// whole number of rows), to the output of the layer being computed, behind the rounds
    /// enqueued so far. `Ok(false)`: the sums of a layer before are still on their way
    /// from page-locked memory; ask again.
    pub fn add_cpu_sums(
        &mut self,
        env: &StepEnv<'_>,
        first: u32,
        sums: &[f32],
    ) -> Result<bool, ModelError> {
        let stream = env.stream;
        let s = self.f.s;
        let n = self.n as usize;
        let (out, device_sums) = (self.at(s.out), self.at(s.cpu_sums));
        let rows = env.record.dims().n_rows as usize;
        let wide = self.wide_state.as_mut().ok_or_else(|| invalid("no wide state"))?;
        if wide.sums_pending && !wide.sums_copied.is_done()? {
            return Ok(false);
        }
        let values = sums.len();
        if !values.is_multiple_of(n)
            || values > wide.cpu_rows as usize * n
            || first as usize + values / n > rows
        {
            return Err(invalid(format!("{values} sums from row {first} of {rows}")));
        }
        let out = out.offset(u64::from(first) * n as u64 * 4);
        // SAFETY: the page-locked buffer and the device's hold `cpu_rows` rows, and the copy
        // that read the page-locked one before has run; the output holds the step's rows.
        unsafe {
            let host = wide.host_sums.as_mut_ptr();
            std::ptr::copy_nonoverlapping(sums.as_ptr().cast::<u8>(), host, values * 4);
            stream.copy_from_host(device_sums, host, values * 4)?;
            wide.sums_copied.record(stream)?;
            self.kernels.add_f32(stream, out, device_sums, values as u32)?;
        }
        wide.sums_pending = true;
        self.mark(stream, Phase::Experts)?;
        Ok(true)
    }

    /// The address of slot `slot` of the pool of layer `i`'s experts.
    pub fn slot_address(&self, i: usize, slot: u32) -> u64 {
        let bytes = self.w.layers[i].moe.shape.slot_bytes().unwrap_or(0);
        self.expert_at[i].pool.as_raw() + u64::from(slot) * bytes
    }

    /// Computes, for the rows of the wide step, the experts of layer `i` that have an
    /// address in `addresses` (one per expert, 0 for none), adding their outputs to the
    /// layer's. The shared expert was enqueued by [`Part::enqueue_layer`]; the first round
    /// makes the layer's plan from the picks (`record`, the router's record as the host has
    /// it, when this part did not route itself). Every expert is computed in one round only:
    /// the host gives each round the experts the earlier rounds did not have.
    pub fn experts_round(
        &mut self,
        env: &StepEnv<'_>,
        i: usize,
        addresses: &[u64],
        record: Option<&[u8]>,
    ) -> Result<(), ModelError> {
        let stream = env.stream;
        let rows = env.record.dims().n_rows;
        let s = self.f.s;
        let shape = self.w.layers[i].moe.shape;
        let bytes = 8 * shape.n_experts as usize;
        if addresses.len() != shape.n_experts as usize {
            return Err(invalid(format!(
                "{} addresses for {} experts",
                addresses.len(),
                bytes / 8
            )));
        }
        let tables = self.at(s.addresses);
        let stride = self.f.addr_stage as usize;
        let (staging, device_table, tile, first) = {
            let wide = self.wide_state.as_mut().ok_or_else(|| invalid("no wide state"))?;
            let layer =
                wide.layer.as_mut().filter(|l| l.index == i).ok_or_else(|| {
                    invalid(format!("layer {i} has no experts waiting for a round"))
                })?;
            let table = layer.rounds as usize % ROUNDS as usize;
            let first = layer.rounds == 0;
            layer.rounds += 1;
            // SAFETY: the staging holds `ROUNDS` tables per layer; the copy of a table that
            // was staged before has run (the host waits for each round before the next but
            // one).
            let staging = unsafe { wide.staging.as_mut_ptr().add(i * stride + table * bytes) };
            (staging, tables.offset((table * bytes) as u64), layer.tile, first)
        };
        // SAFETY: the staging and the device table hold `bytes`; scratch is this part's.
        unsafe {
            std::ptr::copy_nonoverlapping(addresses.as_ptr().cast::<u8>(), staging, bytes);
            stream.copy_from_host(device_table, staging, bytes)?;
            let wide = self.wide_state.as_ref().ok_or_else(|| invalid("no wide state"))?;
            if first {
                let record_at = self.at(s.rec_full);
                if let Some(record) = record {
                    let bytes = record_bytes(&shape, rows) as usize;
                    let record = record.get(..bytes).ok_or_else(|| invalid("a short record"))?;
                    stream.copy_from_host(record_at, record.as_ptr(), bytes)?;
                }
                self.kernels.prepare_prefill_activations(
                    stream,
                    ares_cuda::PrefillPath::F16,
                    self.act_values(s.m, self.n, rows),
                    self.at(s.experts_f16),
                    self.n,
                    rows,
                )?;
                wide.kernels.plan_rows(
                    stream,
                    &shape,
                    rows,
                    tile,
                    record_at,
                    self.at(s.plan_rows),
                    env.token,
                )?;
            }
            wide.kernels.rows_gate_up(
                stream,
                &shape,
                rows,
                tile,
                device_table,
                self.at(s.plan_rows),
                self.at(s.experts_f16),
                self.at(s.hidden_rows),
            )?;
            wide.kernels.rows_down(
                stream,
                &shape,
                rows,
                tile,
                Accumulate::Atomic,
                device_table,
                self.at(s.plan_rows),
                self.at(s.hidden_rows),
                self.at(s.out),
                DevicePtr::NULL,
            )?;
            wide.done.record(stream)?;
        }
        self.mark(stream, Phase::Experts)
    }

    /// Whether the rounds of experts enqueued so far have run on the device.
    pub fn rounds_done(&self) -> Result<bool, ModelError> {
        let wide = self.wide_state.as_ref().ok_or_else(|| invalid("no wide state"))?;
        Ok(wide.done.is_done()?)
    }

    /// How many pairs of the wide step picked each expert of layer `i`, from the layer's
    /// record as the router's board has it (the ids follow the 16 bytes of the header).
    pub fn picked_counts(&self, i: usize, rows: u32, record: &[u8]) -> Vec<u32> {
        let shape = self.w.layers[i].moe.shape;
        let picks = shape.top_k as usize * rows as usize;
        let mut counts = vec![0u32; shape.n_experts as usize];
        for id in record[16..16 + 4 * picks].as_chunks::<4>().0 {
            if let Some(count) = counts.get_mut(u32::from_le_bytes(*id) as usize) {
                *count += 1;
            }
        }
        counts
    }
}
