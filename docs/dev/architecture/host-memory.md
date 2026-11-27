# Host memory

There is one setting for RAM: the most the whole engine may use (`--ram`;
without it, all the machine has). Inside that limit the users of memory make
room for each other; nobody is given a fixed share. The limit only caps what
the engine holds. What it really holds follows the machine: page-locked
memory is the one kind the kernel cannot take back (it is neither swapped nor
reclaimed), so an engine that locks what is free must watch the machine
itself, and it gives memory back in time when another program grows, with or
without a limit that was named.

| User | Kind | When memory is short |
| --- | --- | --- |
| What the engine needs to run: step buffers, read-back buffers, the token embedding table where it lies on the host; the floor of the prompt cache for a model with recurrent state (one checkpoint place, and with `--cache` 16 pages); with `--cache` the buffers that checkpoints pass through on their way to the drive and back (ordinary memory) | Required | Never gives way. A limit below it is refused at load |
| K and V of running sequences beyond VRAM (Flash-Next: the pages from the plan's `near` on, `blocks/far.rs`) | Required, as it is used | Never gives way. It grows in pieces (about 64 MiB over all attention layers) as the sequences reach far pages and shrinks when no page in use lies in a piece; the step that needs a piece makes the tier of experts give it up first (see "K and V beyond VRAM") |
| RAM tier of MoE experts | Elastic, the highest | Gets all that is not required, as far as it can use it, and is the last to shrink: it gives pieces back only when the prompt cache has nothing left in RAM |
| Staging of the experts on their way from the file into transit | Part of the tier | The pieces that hold it are never given up |
| Prompt cache in RAM (checkpoints of recurrent state, the pages that left the devices, and the K and V beyond VRAM of pages that only the prompt cache holds on the devices) | Elastic, the lowest | Goes first: the far pages only it holds leave the devices first (into RAM when its allowance has room, otherwise nowhere: they never wait for the drive); what the devices or the drive also hold goes at once; what only RAM holds is written to the drive first with `--cache` and goes when it is written, down to the floor above; without `--cache` it goes at once, down to its one checkpoint place |

Nothing is reserved for the K and V beyond VRAM: they take their memory when
a sequence reaches them, from the elastic users, as the rank of required
memory says (a running request must never fail to offload its K/V pages
because a cache is sitting on memory it could do without). The pages a
finished request leaves in the device pool as prompt cache rank as prompt
cache with their far K and V (the owner's answer of 2026-11-22): the books
count those bytes in the prompt cache's share and not against the tier
(`PinnedRam::set_context_cached`), and while the prompt cache holds more than
its allowance the conductor takes such pages off the devices first
(`Cache::trim`, `PrefixIndex::evict_where`): into RAM when the allowance has
room, on to the drive with `--cache`, or nowhere. The far pool unlocks a piece
once no page in use lies in it, and the memory goes back to the tier. With a
tier that wants more than the limit leaves (Flash-Next at any `--ram` below the
size of its experts), a long conversation's far pages therefore leave the
devices when its request ends, and its next turn reads them from the drive or
computes them again; its pages in VRAM stay.

The books of the limit count what each user holds: a user asks to grow, a
booking beyond the limit is refused, and the refusal names who holds what.
Below the limit every elastic user keeps to an **allowance**, the bytes it may
hold now, and that is what moves. The rank sets it, in one place
(`Books::allowances` in `crates/runtime/src/ram.rs`): the tier of experts may
hold all the room the required users leave but the prompt cache's floor, the
prompt cache what the tier does not want, at least its floor; an engine that
follows the machine holds each to the share the machine gives it above its
floor as well (the machine shares out by the same rank, see *Dividing*
below). VRAM is handled the same way in spirit but is planned ahead (ledgers),
because moving things there is costly. (Until 2026-11-24 a general broker in
`ares-core`, with priorities, floors and reclaim plans, sat under these books;
the runtime turned every plan into a refusal and kept its own policy on top,
so the broker went.)

**The rule** (`crates/runtime/src/ram/machine.rs`: `Rule`, `Follow`,
`Machine`). One watch per process reads `MemAvailable` every 50 ms, and every
5 ms while a release is under way.

- *The headroom* is what the engine leaves available. It is not a round
  number but the sum of two things that can be stated. First, what the kernel
  keeps free for itself before it stops reclaiming: the high watermarks of
  its zones (`/proc/zoneinfo`; 33 150 pages, 129.5 MiB, on the machine of
  the measurements below). Second, what other programs can take while the
  engine is still giving: the fastest growth of the others that was seen
  (bytes per second over the last four looks, what the process itself holds
  taken out) times the time the engine needs to react (the period of the looks
  plus the slowest release that was seen, from the look that asked to the
  look that found the memory gone). Until something faster or slower was
  seen the rule assumes 2 GiB per second (one thread that writes to new
  pages: the helper of the measurement below wrote 1.4 to 1.9 GiB per
  second) and 100 ms; what was seen above the assumptions is forgotten by
  half every ten minutes. On that machine the headroom is 129.5 MiB + 2
  GiB/s x 150 ms = 437 MiB until something else was seen. It has a most, a
  sixteenth of the machine's RAM (1.95 GiB there), so that one jump of the
  machine's numbers does not cost the engine its memory for the hour it
  takes to forget it.
- *Giving back.* When less than the headroom is available, the elastic users
  together may hold that much less than they hold, at once: the checkpoints
  first, the tier of experts when they have nothing left, the engines of a
  process alike. What is required is not in that sum and is never asked for.
- *Taking.* When more than the headroom and a band (two pieces, or a quarter
  of the headroom when that is more) has been available for a while, the
  users may hold the surplus down to the middle of the band. The wait is 3 s
  at first and doubles, up to 5 minutes, when memory had to be given back
  within a minute of taking it (once for every take); a give long after the
  last take starts it at 3 s again. A program that allocates and frees is so followed down at
  once and up more and more slowly. Memory that is free when a model is
  loaded is taken at the load without the wait.
- *Dividing.* What the elastic users may hold is divided by rank: the tiers
  of experts of all loaded models first, each the same as far as it can use
  it, then the checkpoints with what is left.

**What is built.** The books are `PinnedRam` (`crates/runtime/src/ram.rs`),
shared by the executor and the device threads. An engine with a fixed limit
(`PinnedRam::new`: the tests and tools that set the limit to what they
measure) holds whatever the limit allows; an engine that follows a `Machine`
(`PinnedRam::following`: every engine of the server) has its allowances set
look by look. The users:

- the step record and the read-back buffers of the devices, with the
  page-locked buffers a model's parts hold (required, booked at start:
  `required_ram`, `Model::pinned_ram`); for `qwen35` that includes the token
  embedding table (313 MiB and 521 MiB in the two files of the 27B that were
  measured) and the staging buffer of the rows a step uploads from it, unless
  the device keeps the whole table (see "The token embedding of `qwen35`",
  also for what a limit below it does: the server refuses it and names the
  shares);
- the tier of experts of a mixture-of-experts model. Its memory is one range
  of addresses planned for all the tier can ever hold (what the limit leaves
  and the experts without a VRAM slot can use), page-locked in **pieces of 64
  MiB** (`expert_store/region.rs`): a piece takes memory when it is locked
  and gives it back to the system when it is released (unlocked, then
  `MADV_DONTNEED`). A piece is whole slots
  of one pool and starts on a page, because a device copies from one locked
  range and not across two. The size comes from what a lock and a release
  cost (RTX 4070, 2026-11-15, load average 22, two rounds each): locking a
  GiB for a device (`cuMemHostRegister`, which also brings the pages in) took
  380 to 680 ms whatever the size of the pieces, 32 to 43 ms for a piece of
  64 MiB; releasing a GiB took 105 to 111 ms in pieces of 64 MiB (7 ms a
  piece, the slowest 10 to 13 ms), 142 to 166 ms in pieces of 16 MiB and 76
  to 102 ms in pieces of 256 MiB (21 to 43 ms a piece). With 64 MiB a release
  takes effect within about 10 ms and a GiB is back in a ninth of a second.
  The kernel's count of the process's resident memory went from 192 to 1216
  MiB and back to 192 in every round. The tier starts with the pieces its allowance
  holds. When the allowance falls, pieces are given up between steps, once no
  read, copy or CPU job touches the tier's memory (the end of a step waits up
  to 20 ms for what is on its way and tries again after the next step
  otherwise; an idle engine gives them up on the watch's thread). A piece is
  a range of addresses whose slots belong to a few layers and hold whatever
  was read into them, so it is not given up as it lies: every layer of the
  pool gives slots in proportion to what it has above the two a layer keeps,
  each layer gives its coldest (the copies of experts that VRAM holds first),
  and the experts that lie in the piece and stay are copied into the slots
  that were given (`expert_store/elastic.rs`; the bookkeeping, `Tier::vacate`,
  is the one the loans of VRAM to a step's scratch use). The driver's unlock waits for
  the device to finish what it has, and a step's kernels can wait on the
  device for an expert the store brings, so a piece is never unlocked while
  a step runs ("When page-locked memory changes" below): what the step's end
  gave up is unlocked once every device has ended the step, by the conductor
  as it collects the step, and the watch's thread does it for an engine that
  runs nothing, with the store held so that no step begins meanwhile. A piece
  that was given up keeps its lock until then and
  is not locked again before (`HostRegion::is_releasing`): the driver
  refuses a lock of a range that a lock still covers
  (`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`). The end of a step that gave
  up a small piece and then a larger one of another pool did that once,
  when the small one fit in what the larger one left below the allowance
  (Flash-Next's pools have slots of three sizes, and the last piece of a
  pool is smaller). When the allowance rises, a piece is locked before
  every step on the conductor's thread (`Flow::before_step`; that step
  begins 30 to 70 ms later; until 2026-11-24 the end of the step before did
  it on a device thread), or by the watch for 40
  ms at a time while nothing runs, and its slots come back empty; they fill
  from what the routers pick, and from the heat while nothing runs (see "The
  usage profile of the experts"), which reads nothing into the tier while it
  holds more than it may. The piece that holds staging is never given
  up, and the tier says so when it can give no more;
- the prompt cache in RAM, one user in the books (`PinnedRam::want_cache`,
  `cache_allowance`, `on_cache`): its pages, a buffer per device share of a
  page that the cache locks when a page leaves the devices
  (`PinnedRam::cache_buffer`, `cuMemHostAlloc` for that device; kept for the
  next page and given back when the allowance falls), and the checkpoints of
  recurrent state, locked for each device part in chunks of at least 32 MiB
  when a place is first used (`PlaceChunks`), before the step or transfer that
  uses it is handed to the devices (`PinnedRam::hold_places`). The conductor keeps no more
  than the allowance holds: it takes nothing new when the allowance has no
  room, gives up what goes for nothing first when it falls and writes what
  only RAM holds before that goes (between steps; an idle conductor is woken
  for it; see "In RAM" under "Sequence state"), and tells the books which
  places are in use. A chunk in which no place
  is in use is unlocked and freed (`PinnedRam::sweep_checkpoints`), and
  locked again when a place in it is used again. In a model whose checkpoint
  is larger than a chunk (the 27B), a chunk is a place; in a small model a
  chunk goes only when all its places are free, and checkpoints are not moved
  to make one free. The allowance has a floor of one place (for a model with
  recurrent state), with the prompt cache on SSD one place and 16 pages, and
  there the conductor gives checkpoints up only after they left their places
  for the drive (see "RAM" under "On SSD").
  The cache counts the floor as one place and the pages; the first chunk of
  every device part holds place 0 alone, so the floor is as large as what its
  place locks even where a chunk holds several places (a small model);
- the buffers that checkpoints pass through on their way to the drive and
  back (`--cache`): ordinary memory, aligned for direct I/O, allocated when
  first used and kept. They are not in the books; the server takes their
  most (`SsdConfig::transit_ram`) off the limit before it hands the rest to
  the books.

Both elastic users meet in the books of a Flash-Next engine: the tier of
experts, and the prompt cache of its recurrent state (`Model::prefix_cache`,
and with `--cache` its level on SSD), whose floor of one checkpoint place the
server books before the tier (`load/ram.rs`, `Budget`); the ranks above decide between
them. A dense model has no tier. A prompt's working memory, staging and
checkpoints do not borrow from each other at run time (staging is a fixed
part of the tier, sized at load for the tier as it starts). The ranks also
decide between the engines of a server (see "Several models"). What meets
inside the prompt cache is the checkpoints, the pages in RAM and the level on
SSD, which reads checkpoints out of the places and writes them back in; the
rules for that are under "In RAM" and "On SSD".

The limit is `GpuConfig::ram_limit`, which the server sets from `--ram`
(default: all the machine has; the log states it, what the machine has
available and the headroom at that moment), less the most the buffers of
checkpoints in transit to the drive take when `--cache` is given, since those
are host memory of the engine that the books do not hold. For a model with a
prefix cache, what the limit leaves after the required memory is the budget
of the prompt cache in RAM (`RuntimeConfig::cache_ram`), so the places and
pages the conductor takes never lock more than the limit; for a mixture of
experts it is what the tier is planned for. A dense model's server therefore
lets the prompt cache use what the machine has, less the headroom, and gives
the coldest up when the machine needs the memory. A lock that does not fit the
limit fails the step with the books in the message. The first use of a place
waits for the lock before the step, on the conductor's thread (21 to 22.5 ms for 93.5 MiB, 11.8
ms for 56.1 MiB, measured with the 27B on the device's thread, where it was until
2026-11-24); locking ahead in the background is
not built. `GET /metrics` reports the bytes locked
(`ares_checkpoint_ram_bytes`, `ares_expert_ram_tier_bytes`) and the counters
of the expert store (`ares_expert_*`).

What these runs also showed, and what follows from it:

- The driver's unlock of page-locked memory waits for the device. The first
  version unlocked on the watch's thread while a step ran, and a load under
  memory pressure stood still for ten minutes, every thread waiting; pieces
  are since unlocked and locked between steps only. A release therefore
  takes as long as the step that runs: 11 ms on two cards that decode, three
  seconds on one card under load, 17 to 25 s during the long steps of a load.
- That time is in the headroom by the rule, so after a load the headroom is
  at its most (a sixteenth of the machine's RAM) and falls back as the slow
  release is forgotten, by half every ten minutes. A release that does not
  wait for a step would keep it at a few hundred MiB; it would need memory
  that the device does not read during a step, and is not built.
- Not tried: a mixture of experts and a dense model loaded on the same
  device, with memory short. Until 2026-11-24 the chunks of checkpoints were
  freed on the watch's thread whatever the devices did, which hung Flash-Next
  itself (below); the rule that every engine of the process shares since then
  frees them between the steps of all of them.
- With `--ram 28G` on this machine the engine holds what is available less
  the headroom when it loads, not 28 GiB: 20 to 21 GiB with the 21 to 22 GiB
  that were available in the calm runs here, of which Flash-Next on both
  cards can use about 16 GiB (its experts without a slot in VRAM) and on one card
  all of it.

**When page-locked memory changes** (`crates/cuda/src/gate.rs`, since
2026-11-24). Allocating, freeing, locking and unlocking page-locked memory
(`cuMemHostAlloc`, `cuMemFreeHost`, `cuMemHostRegister`, `cuMemHostUnregister`)
take a lock of the driver's that every call of the process needs, and a free
or an unlock waits under it for the work the devices have queued. NVIDIA
documents part of it: the CUDA C Programming Guide lists "a page-locked host
memory allocation" among the operations between which the commands of two
streams cannot run at the same time (3.2.6.5.4, "Implicit Synchronization"),
and the driver's `cuMemFree` "may perform implicit synchronization"; PyTorch's
host allocator notes that `cudaFreeHost` "waits for the work already queued on
the current device" (`aten/src/ATen/core/CachingHostAllocator.h` at 09daa3ff)
and frees outside its locks. llama.cpp frees page-locked memory only when a
buffer goes (unload) and vLLM takes its page-locked buffers once; neither
changes it while it serves, so there was no code to take. A step's kernels
wait on the device for the host: for the slot of an expert the store brings
(`expert_ffn.cuh` polls the table entry that `CudaSlots::set_entry` copies),
for the host's claim and sums of an expert the CPU computes, for the peer's
half of a layer. The thread that ends such a wait enqueues a copy, and the
copy needs the driver's lock. A free on another thread while a step runs
therefore waits for the kernel, the kernel for the thread and the thread for
the free, for good (a kernel gives up after 0xF0000000 polls, about an hour).
Seen 2026-11-24 with Flash-Next on both cards at `--ram 8G` and a named
context beyond VRAM (`real_cache`'s resume test; two of three runs of its
requests alone): the watch's thread in `cuMemFreeHost` of a chunk of
checkpoint places (`PinnedRam::sweep_checkpoints`), device 1 in
`cuMemcpyHtoDAsync` of a slot entry, waiting for the driver's lock, the
conductor waiting for device 1, the engine deaf to SIGINT. The rule, for every
thread of the process: **page-locked memory changes only while no device runs
a job** (a step, a transfer between the devices and the host, a job of the
image tower). The executor holds the gate from the hand-over of a job to its
end (`gate::Steps`, from `GpuExecutor::dispatch` to `collect`; the hand-driven
`Session` of the model's tests around its step), and each device thread marks
itself inside the job (`gate::InStep`). Then:

- A lock or an allocation (`gate::change`, in `PinnedBuffer::new` and
  `HostRegistration::new`) waits until no job runs and keeps the next one
  from beginning until it is done; inside a job it fails with an error
  instead of waiting for the end of its own step. A thread that holds what a
  step waits for (the watch's thread with the expert store held) only tries
  (`gate::try_change`).
- A free or an unlock (`gate::retire`, in the drops of `PinnedBuffer` and
  `HostRegistration`) is done at once when no job runs, and otherwise when
  the last job that runs ends, on the thread that ends it (the conductor,
  in `collect`, before the next step). A user that books its bytes (a chunk
  of checkpoint places, a page of the prompt cache in RAM, a piece of the
  tier of experts or of a far pool) retires the memory together with its
  booking and with what follows the unlock (`MADV_DONTNEED`, the piece free
  to be locked again), so the books give the bytes back when the memory is
  free, and a piece is not locked again before its unlock is done.
- So what a step needs locked is locked before it is handed over, on the
  conductor's thread: the chunks of the checkpoint places that the record's
  copies and the saves inside the step use (`PinnedRam::hold_places`, also
  before a transfer that brings checkpoints in), and in Flash-Next the pieces
  of the far pools that hold the far pages the record's tables name and a
  piece more of the tier of experts when it may hold more
  (`Model::before_step`, `Flow::before_step`; a wrapper of a model must pass
  the hook on, and a step with far pools that was not prepared so fails). The
  buffers that a prompt read layer by layer reads its picks into and sends its
  inbox from are taken when a part is opened.

The gate is the process's and not an engine's, as the driver's lock is: an
engine that loads beside one that serves waits for its steps, and what an
engine frees while another runs a step waits for that step's end. What the
frees between steps cost is measured in
[`../experiments/host-memory.md`](../experiments/host-memory.md) ("Page-locked
memory between steps"). Device memory is freed behind the gate as well (since
2026-11-25): `cuMemFree` "may perform implicit synchronization" too, so a
`DeviceBuffer` dropped inside a step or beside another engine's step is freed
where no job runs (`gate::retire`). A step allocates no device memory and makes
no blocking copy either (since 2026-11-26, GitHub issue #2): the 27B's part
keeps room for the largest attention plans its steps can make
(`PlanConfig::plan_bytes_bound`, per group of rows and number of KV heads),
and a plan that does not fit fails the step instead of taking `Plan::upload`,
whose allocation and synchronous copy can wait for a kernel that waits for
the other device's thread. An engine takes most of it when its model opens, and
the frees of one that unloads beside another that serves wait for that one's
step; what looks at the free memory of a device (`Device::mem_info`, which a
placement reads) or finds no room for an allocation (`DeviceBuffer::new`)
first waits for the frees that are due (`gate::settle`), so that a model loaded
after an unload sees the memory the unload gave back.

Nothing that costs speed happens silently:

- **At load**, the engine states what the configured context means: how many
  tokens of state fit in VRAM, and, if the context is larger than that, that
  K/V will spill to RAM, from which point, and how much RAM that can take at
  most. A context that fits entirely in VRAM never offloads. It also states
  the RAM limit, what the machine has available and the headroom.
- **When the expert tier in RAM shrinks**, generation gets slower: experts
  that were in RAM now come from the SSD. The tier's bytes are a gauge, and
  the pieces given up and taken are counted (`StoreStats::pieces_released`,
  `pieces_locked`, `piece_evictions`), and the server writes a line in the
  log for every shrink: what was given back, what the tier holds and may hold
  (`ExpertStore::on_shrink`, `Qwen4Exp::with_log`).
- **When the prompt cache in RAM shrinks**, that is routine and only counted
  (`ares_checkpoints_dropped_total`, `ares_cache_ram_pages`,
  `ares_cache_ram_bytes`). With `--cache` what only RAM holds goes to the
  drive before its memory goes; what was due there and was given up
  unwritten is counted apart (`ares_cache_ssd_checkpoints_skipped_total`,
  `ares_cache_ram_pages_dropped_total`).

The same facts are available as counters to whoever monitors the server.

## Loading and unloading

Switching models must take seconds, bound by the hardware and not by the
engine. What that needs:

- **Nothing to start.** The server process, the device contexts and the
  kernels stay loaded while models come and go. The CUDA kernels are
  precompiled machine code; the Vulkan pipelines want a pipeline cache on
  disk. (Today the server process stays, and a device's driver context
  stays while a model is on the device: it is opened with the first model
  there and closed with the last, so that a server with nothing loaded holds
  no device memory and a model whose device failed is loaded again on a
  fresh context. A model that takes the place of the only other one on its
  devices therefore opens them again; what that costs is part of its load
  time and is not measured apart.)
- **Weights at the speed of the slowest link.** Tensors are read with direct
  I/O (or taken from the page cache when the file is already there) into
  page-locked staging and copied to all devices at once, each device on its
  own stream. The limits are then the SSD and each card's PCIe link, about
  3.3 GB/s for the RTX 3060 on its x4 slot.
- **State is reserved, not prepared.** The page pool is a few large device
  allocations. Pages are never cleared, because a page table decides what is
  valid, and there are no warm-up passes, because there are no graphs to
  build. Page-locked host memory, which is slow to lock, is taken in chunks
  in the background and only as far as offload needs it. (Today the only such
  memory is the checkpoints' places, which a device thread locks in chunks
  when one is first used, not in the background; see "Host memory".)
- **Unloading frees and returns.** With `--cache`, what is worth keeping of
  the prompt cache is written at unload: what only RAM holds and the pages
  and checkpoints of the devices (see "On SSD" under "Sequence state"), and
  a model that is loaded again finds its conversations in its store; without
  it the prefix cache of a model goes with it. A mixture of experts writes
  the usage profile of its experts (96 KiB for Flash-Next) if steps ran
  since it last did. The
  unload ends the engine's requests, joins its threads, drops the engine and
  compares what every device reports as free before and after with what the
  model took at its load; the log has both and the time.

Models are loaded and unloaded while the server runs by its router mode (see
"Several models" under Server). Load and unload times of the real files
through it are not measured yet.
