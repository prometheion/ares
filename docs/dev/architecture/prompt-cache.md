# The prompt cache in RAM and on SSD

**In RAM.** The prefix cache has a level in host memory below the devices,
with or without a level on SSD under it, for every model whose device side
moves its state to the host and back (`Model::durable`,
`DeviceModel::transfer`; both models do, on the shared slot block).
The levels are the devices, RAM and the drive, as in the owner's llama.cpp
fork; there is no level of the prompt cache in VRAM besides the working state
on the cards. What RAM holds and what goes first is
`crates/runtime/src/cache/ram.rs`; when things move, `cache/moves.rs` (to and from
the drive `cache/ssd/pump.rs`). A run that comes back from either level enters the
index through the same helpers (`run_of`, `deepest`, `run_pages`, `take_page`,
`keeps_form`, `Cache::hang`), RAM's at once inside the look, the drive's across the
pump's rounds.

- **Who takes part.** Every request, unless it says `cache_prompt: false`
  (`GenerateRequest::cache`): such a sequence asks the cache nothing when it
  arrives or starts, publishes no pages, tails or checkpoints, and its pages go
  back to the pool with it; the conductor skips the cache for it wherever it
  would look at its chain (`Active::cache`). The parts of a decision use the
  cache; what a fork computes does not enter it (`Active::enters_cache`, see
  "Decisions" in [sequence-state.md](sequence-state.md)).
- **What it holds.** A page or tail that the index gives up is copied out of
  the pool into RAM before anything writes its page of the pool again
  (`Cache::settle_leaving`, which runs before every record and at the front
  of every batch of transfers), under the name the index gave it; its
  checkpoint goes with it and stays in its place. A copy is a buffer per
  device share of the page, page-locked for the device that copies it
  (`PinnedRam::cache_buffer`), so the copies are DMA both ways.
- **A hit.** A prompt that arrives, and a waiting one that is looked at
  again, has what RAM holds after the pages of the index copied back before
  the step is planned (`Cache::probe` with the executor): the pages take
  pages of the pool (cached pages nobody holds make room, and go to RAM
  first, in the same batch), they enter the index in order, and the
  checkpoint that hung on one of them is attached to it, in its place. A
  tail whose tokens the prompt goes on with comes along with the state at
  its end, so the next turn of a conversation replays nothing. For a model
  with recurrent state the run ends at its deepest checkpoint, unless a tail
  with its state follows or the drive goes on to a deeper one. RAM keeps its
  copies: a page that comes back and leaves the devices again costs no copy.
  A request that the accounting starts in the middle of a plan takes what the
  index has; only an arrival and a look before the plan bring pages back,
  since they take pages of the pool that the plan must see. A waiting
  request brings pages back once (and once more after it gave way): two
  requests that wait would otherwise take each other's pages at every step.
- **What goes first.** By class, then the least recently used: copies of
  pages that the devices hold; what the drive holds and what is not worth
  writing (a text shorter than the level on SSD's least, a tail without the
  state at its end, a checkpoint that later requests have left behind at
  deeper page boundaries), and everything when there is no level on SSD; then
  the rest, which is written first. A page is
  the most recently used of what left the devices when it arrives, and the
  devices give a text up from its end, so a text goes from its end here too.
  A checkpoint goes on its own or with its page. When the cache gives memory
  back down to its floor, the index's checkpoint used last (the deepest of
  them) goes last, whatever the ranking says (`give_up_toward_floor`,
  2026-11-25): the floor of one place is there so
  that a conversation goes on from its last checkpoint, and the ranking, which
  keeps roots longest, kept the root of a long prompt and gave up its anchor,
  so the same prompt sent again replayed every row the cache held.
- **Memory.** The pages and the checkpoints are the prompt cache's share of
  the engine's page-locked RAM, its lowest user ("Host memory"): the cache
  takes what the allowance leaves and gives back what it does not hold any
  more (`Cache::trim`): what goes for nothing first, then what only RAM
  holds, which is written and goes when the writer is done with it, then the
  coldest checkpoints of the index. Checkpoints come before pages: a new
  checkpoint makes RAM give up what goes for nothing, then the checkpoints
  that hang on pages in RAM, then the index's by its ranking; pages above
  their floor make room for checkpoints, not the other way round. The
  devices lock a place's memory when a step first uses it, and a lock beyond
  `--ram` would fail the step, so a checkpoint takes a free place whose
  memory is still locked when there is one, else a place whose lock the
  limit has room for (`PinnedRam::place_lock`), after giving up what goes for
  nothing; with neither it is not taken. Until the step with its save has
  run, that room is the place's: a new buffer of pages leaves it free.
  (Measured: without this a 29k conversation's pages in RAM left the next
  turn's checkpoint no room under `--ram 3G`, and the step failed.)
- **Room for what comes.** When RAM is full, the coldest of what only it
  holds is written ahead, so that there is room for 16 pages and a
  checkpoint without waiting for the drive (`Cache::clean_ahead`). With a level on SSD the cache keeps
  one checkpoint place and 16 pages whatever the machine has, its floor in
  the books: state passes through them to the drive and back, so a prompt
  resumes from SSD while the machine has no room for the cache, and a page
  that leaves the devices reaches the drive through them. Such a page waits
  for the writer when nothing else gives room (`Cache::page_room`), and
  reads from the drive take half of the floor at most, so that pages that
  leave the devices for them find the other half.
- **Counted** in `/metrics`: `ares_cache_ram_pages`, `ares_cache_ram_bytes`,
  `ares_cache_ram_pages_in_total`, `ares_cache_ram_pages_read_total`,
  `ares_cache_ram_pages_dropped_total` (pages worth keeping that went
  unwritten: no room, or no level on SSD), and for both levels at once
  `ares_cache_transfer_nanoseconds_total` (the conductor's wait for the
  batches of transfers, counted once whichever level a batch serves).
- **Measured** 2026-11-16 with the server: Qwen3.8-27B Q4_K_S on the RTX 4070
  and the RTX 3060, tensor split 0.625 / 0.375, two sequence slots, the
  model's MTP block on, `--ram 3G`, `--context 32768` (so that another
  conversation of 29 322 tokens pushes the first one out of the cards), and
  `--cache` on the btrfs volume (compressed, shared with other work). A
  conversation A of about 8k or 30k tokens (a list of facts and a
  question): turn 1 cold, turn 2 from the cards, then a conversation B of
  29 322 tokens, turn 3 of A, then the server stops and starts again, and
  turn 4 of A. Time to the first token of a stream on `/v1/completions`,
  most probable token; the load average next to each run. Before is the
  build without the level in RAM (pages written as they were computed), in
  whose turn 3 A comes back from the drive.

  | First token after | 8k, after | 8k, before | 30k, after | 30k, before |
  | --- | --- | --- | --- | --- |
  | Computed (turn 1) | 6.33 s (load 25) | 6.23 s (load 5) | 30.15 s (load 4); 30.14 s (load 8) | 30.31 s (load 3) |
  | From the cards (turn 2) | 0.125 s | 0.131 s | 0.136 s; 0.133 s | 0.142 s |
  | After B pushed A out (turn 3) | 0.262 s from RAM (33 pages) | 0.302 s from SSD (32 pages); 1.815 s with the drive busy (load 12) | 0.461 s from RAM (202 pages, load 3); 1.327 s (load 21) | 0.532 s from SSD (201 pages) |
  | After a restart, from SSD (turn 4) | 0.943 s (60 pages, load 17) | 0.596 s (load 4); 1.152 s (load 12) | 0.839 s (229 pages, load 2); 2.62 s (load 15) | 0.673 s (load 2) |

  Nothing was replayed in any turn that resumed. Turn 3 from RAM has to
  move B out of the cards first (into RAM, a copy of its 203 pages and new
  buffers for them), where before B's pages were on the drive already and
  went for nothing; with the drive busy, the read from SSD before took
  seconds. Of the 1.327 s at load 21 the copies took 0.236 s both ways; the
  rest is mostly locking new buffers for B's pages (RAM grew by 0.9 GB), a
  cost of the first time RAM grows that far, which locking ahead would
  hide. B's pages moving into RAM while B was read cost 0.27 s of its 33 s. What was written: before, every turn wrote what it computed
  (8k: 430 MB in turn 1, 5 MB in turn 2, 1214 MB with B; 30k: 1214 MB in
  turn 1 and 1214 MB with B); after, nothing was written while the server
  ran, and the stop wrote what was worth keeping: 1.65 GB in 17.9 s (8k
  case, load 19) and 2.43 GB in 8.8 s and 28.4 s (30k case, loads 3 and
  21), the pages of A and B and their checkpoints at page boundaries. A stop before writes only
  what was still queued (0.4 s).

**On SSD.** The prefix cache has a level on SSD below RAM (`ares --cache
<dir>`; `crates/runtime/src/cache/ssd/` over `ares-store`). What leaves RAM
is written to a store, and so is what is worth keeping when an engine ends; a
prompt whose start is in none of the levels above but in the store has it
brought back before it starts. A conversation therefore resumes after its
pages left the devices and RAM, after its model was unloaded (router mode),
and after a restart of the server. Nothing is written as it is computed.

- **The store** (`ares-store`) keeps two kinds of records under the chain
  hash of a page, the page and the checkpoint of recurrent state at its end,
  in a directory per fingerprint. A record is a header block and one part per
  device, each padded to 4096 bytes, so parts are written from and read into
  aligned buffers with direct I/O (through the page cache on a filesystem
  without it). Pages and checkpoints lie in segment files of their own (256
  MiB): a conversation's pages sit together, and its checkpoints, which later
  turns leave behind, age and go without taking pages along. The index is
  rebuilt from the record headers when the store is opened. One process uses
  a store at a time (a lock file); a second server on the same directory and
  fingerprint runs without the level and says so, after waiting a second for
  the lock: the engine before may hold it a moment longer, and so does a
  process forked from this one until it runs its program (the tests' full
  drive, which made two cache tests of an engine after a restart fail now
  and then, their store "in use by another process").
- **Bytes move outside of steps.** A page is written straight from its
  buffer in RAM by one writer thread, which syncs the drive a second after its
  last record (and every ten seconds while records keep coming); the buffer
  goes back to RAM, whose page is then on the drive too. A checkpoint is
  copied out of its place into a buffer of its own first (three of them,
  ordinary memory), by the device threads (`Executor::transfer`), before its
  place can be written again or lose its memory. Four reader threads read
  what comes back into buffers of RAM (half the floor of pages in flight at
  most) and check it; the conductor moves it to the devices between steps
  (`Cache::pump`, at the top of its loop). The step loop does not wait for
  the drive, but in two cases: a checkpoint that must leave its place when
  the writer holds every buffer (up to 250 ms, then it is given up and
  counted), and a page worth keeping that leaves the devices when RAM has
  nothing else to give room (it waits for the writer to finish a page). The
  store's index is locked to reserve or find a record, never while bytes
  move.
- **What is written when.** A page when it leaves RAM, or ahead of that when
  RAM is full, if it is worth writing: its sequence held 2048 settled cells
  at some point (`SsdConfig::min_cells`; below that a text is computed again
  in about a second and a half on the 27B, and what would be written is
  mostly its checkpoint, which is as large as the pages of several thousand
  cells). Then all its pages are worth writing, from the first on, and a page
  that is written makes the one before it worth writing, so that a text on
  the drive is found from its start. A checkpoint at a page boundary at least
  that deep when it leaves RAM, unless later requests have left it behind at
  deeper page boundaries (`PrefixIndex::left_behind_on_pages`: a deeper
  checkpoint at a page boundary on its one branch is where a prompt resumes
  from the drive). When an engine
  ends (a clean stop, an unload in router mode) everything worth writing that
  the drive does not have is written, the checkpoints first, then what only
  RAM holds, the coldest first, then the pages of the devices; the engine
  waits for the writer up to two minutes. A record the store has is not
  written again. A tail with the state at its end (a prompt's anchor at the
  end of its last message, a request's end) is written like a page: a page
  record under its tail hash, one deeper than the page it hangs under, and a
  state record beside it; a tail whose state has gone is not. The drain
  skips a state whose save still waits for a record, which at the end never
  runs: a request's end state is saved with the next record, so the last
  request's end is not written, and its next turn after a restart resumes at
  its prompt's anchor and computes the reply's rows once.
- **A hit.** `Cache::probe` asks the store for the pages after the ones the
  index (and RAM) have. A page of the prompt that RAM has handed the writer
  is in neither (its bytes are with the writer, and the store has no record
  yet): the probe waits for the writer to give such pages back first, at
  most 250 ms, and RAM brings them back (`Cache::await_transit`). For a model with recurrent state the run ends at its
  deepest checkpoint in the store (the pages behind it would be replayed row
  by row, which takes as long as computing them), and without a checkpoint
  nothing is brought back. The deepest may be the state at the end of a
  stored tail the prompt goes on from, under a page of the run: the store
  lists the pages under each parent (`PageStore::children`), and only a page
  that has one besides the next page of the text is tried, every length of
  the prompt's cells after it in one pass (`ChainHash::extensions`). The run
  is then the pages before the tail and the tail, which enters the index as
  one with its state; a tail the index holds already is passed over, and a
  prompt that finds one a run brings back waits for it. The run is announced in the index under an owner
  of its own, so the prompt, and any other prompt that shares it, is held as
  for a page that another sequence computes. The readers bring the pages and
  the checkpoint into host memory, the conductor gives each page a page of
  the pool (cached pages nobody holds make room), the devices take the bytes,
  and the pages enter the index in order, the last one with the checkpoint,
  which went into a checkpoint place; the buffers stay in RAM as copies. The
  prompt then starts as from any cached prefix. When the checkpoint finds no
  place, what RAM holds goes for it, the first to go first and without a write
  (`Cache::import_place`): the prompt that waits outranks texts that may come
  again. A record that cannot be read or fails its checksum, a page of the
  pool or a place that cannot be had end the run (it is cut), and the next look
  for that run does not ask the store again (`Ssd::cut`): the prompt computes
  from what the cache has, and a later look tries again. Until 2026-11-28 a cut
  run was asked for again by the very next look, and a prompt whose run never
  found a place waited for ever while the store read it about 17 times a
  second (#98).
- **Forms.** With `q7` to `q4`, and for the MTP block always, a page is an
  exact entry while it is the first tile or among the newest cells and a
  converted one later. A page goes to RAM, and from there to the store, in
  the form it has when it leaves the devices, with a tag; the pages a request
  leaves behind its end are exact entries. A page that comes back exact takes
  an exact slot when it is the first tile or lies at the end of the run, and
  is converted on the way in otherwise, as the window of a sequence that went
  on would have converted it. A tail comes back as the partly used page it
  is: it keeps its exact slot until it is freed and is never converted for
  room (a prompt that continues it copies its leading cells, which only an
  exact entry holds).
- **The fingerprint** covers everything besides the tokens that changes the
  stored bytes: the model's files (their size, their first 64 KiB and 16
  samples of 4 KiB spread over each), the formats of K and of V, the draft
  block and its formats when the model drafts, the format of the recurrent
  state, how the products round (exact products, a tool of the tests, are
  another store), the page size, and which
  heads of every block each device holds (the bytes are stored per device, so
  another split or other shares are another store). The names of the pages
  start from the fingerprint, not from the model's id, so a model keeps its
  store under another name. State of another fingerprint is never read; it
  counts against the size limit and is the first to go.
- **The size limit** (`--cache-size`, 32 GiB by default) is for everything in
  the directory. When it is passed, stores of other fingerprints that no
  process has open go first, then segments none of whose records a prompt can
  still reach (a page is reachable while every page before it in its chain is
  stored, a checkpoint while its page is), then the segment used least
  recently; the order of use survives a restart in the files' modification
  times. Several models that are loaded at once keep within the limit each.
- **Failure.** The level is a cache and never reaches a request: a record
  that cannot be written is dropped and counted, a full drive gives up old
  segments and the write is tried once more, a directory that vanished is made
  again (what was written between its loss and the next sync is lost), and a
  damaged or missing file costs the runs that needed it.
- **RAM.** The level on SSD lies below the one in RAM, which follows what the
  machine has ("Host memory"), and this is how the two meet:
  - *A floor is kept whatever the machine has*: one checkpoint place and 16
    pages (`PinnedRam::want_cache`), as the staging of the experts is the
    floor of their tier: the machine is not given it, and a tier of experts
    could not take it. On a machine with no room for the prompt cache a
    prompt still resumes from SSD (its checkpoint lands in that place and is
    restored from it, its pages pass through the floor), and what leaves the
    devices still reaches the drive, through the floor and at the drive's
    pace. Without `--cache` the floor is the one checkpoint place (for a model
    with recurrent state; the owner's decision of 2026-11-16): the same machine
    leaves the cache that place in RAM and no page, so a conversation still
    resumes from its last checkpoint instead of replaying every row it found.
  - *A checkpoint leaves its place before the place is called unused*: the
    trim copies what it gives up out of the places first, and a place whose
    copy has not run keeps its memory. A device part that is asked to read a
    checkpoint out of a place whose memory went back fails that transfer
    instead of locking new memory to read from (`PlaceChunks::held`).
  - *A save that has not run has nothing to write.* A checkpoint that is
    given up before the record with its save was written (memory ran short
    between two steps, or a checkpoint from the drive needed the place)
    loses that save, and the level on SSD forgets it with it: nothing is
    copied out of a place that never held it. The next checkpoint of the
    sequence, or of the next turn, is the one a later prompt resumes from.
  - *Nothing of this is silent.* The checkpoints that went from RAM are
    counted (`ares_checkpoints_dropped_total`), the ones that were due on the
    drive and were not written (`ares_cache_ssd_checkpoints_skipped_total`),
    and pages worth keeping that went unwritten
    (`ares_cache_ram_pages_dropped_total`).
  - *The buffers of checkpoints in transit* (3 at their most, 471 MB with the
    27B: 3 x 157 MB) are ordinary memory outside the books of page-locked
    memory. The server counts their most against `--ram` next to those books:
    a limit below what the engine needs to run with them and the floor is
    refused, and the plan line of the log names them and the floor. The
    watch over the machine sees them as memory of its own process, not as
    another program that grows.
- **Measured before the level in RAM** (pages were written as they were
  computed, through buffers of ordinary memory) 2026-11-15 with the server and
  `--cache`: Qwen3.8-27B Q4_K_S
  on the RTX 4070 and the RTX 3060, tensor split 0.625 / 0.375, two sequence
  slots, the model's MTP block on. A conversation whose first prompt has
  29 322 tokens (a list of 1900 lines and a question) and whose second turn
  adds the reply and another question (29 354 tokens); requests on
  `/v1/completions`, most probable token, time to the first token of the
  stream. The store lies on the btrfs volume (an NVMe drive, compressed with
  zstd) that the model files are read from, and other work used the machine
  and that drive throughout: the load average is given with every time, and
  `dd` with direct I/O and blocks of 4 MiB wrote 1.0 to 1.2 GB/s and read 1.9
  GB/s when the drive was free, and read 160 to 730 MB/s when it was not.

  | First token after | K and V in `q8` | K and V in `q5` |
  | --- | --- | --- |
  | Computed (nothing cached) | 30.39 s (load 11); without `--cache` 30.73 s (load 17) | 34.77 s (load 19) |
  | Second turn, from the devices | 0.150 s (load 11); without `--cache` 0.144 s | 0.191 s (load 19) |
  | Second turn after the pool gave up 92 of its 229 pages and their checkpoint | 0.59 s (load 23) | not measured: the pool of `q5` held the conversation |
  | Second turn in a new process, all 229 pages and the checkpoint from SSD | 1.15 s (load 17); 2.1 to 3.0 s in four more runs (load 12 to 40, the drive busy) | 2.90 s (load 18, the drive busy) |

  The replies were the same in every row. What the store took: with `q8` a
  page of 128 cells is 4.62 MB (36.1 MB per 1000 tokens), with `q5` 3.17 MB
  on average (24.8 MB per 1000 tokens; the pages a request leaves behind its
  end are written as exact entries, and the first page always), and a
  checkpoint is 157 MB, one for every turn whose prompt ends on a new page.
  The first turn with `q8` wrote 229 pages and a checkpoint, 1.21 GB, in 1.28
  s of the writer's time (0.95 GB/s); over a whole run the writer needed 44 s
  for 3.9 GB, since a sync took seconds whenever other work wrote to the
  volume. The step loop does not wait for the writer: reading the prompt
  took as long with the store as without it, and of that time the loop spent
  0.30 s in transfers (1.0 %). A request with a prompt of 2970 tokens that
  then writes 600 tokens (27 pages and a checkpoint to the store) took 5.27 s
  with the store and 5.01 s without it, eight runs each at loads of 14 to
  21; the transfers were 0.06 to 0.32 s of it, and its tokens came at 218
  per second against 233. A page costs a transfer of 3 to 5 ms and a
  checkpoint one of 30 to 60 ms, each once, so a long generation pays about
  one percent. In the run of 1.15 s the four readers read the 1.21 GB in
  0.89 s each (1.36 GB/s together, against the 1.9 GB/s of `dd`), the
  devices took the bytes in 0.25 s, and the rest is the 42 rows after the
  checkpoint; with the drive busy a reader needed 1.0 to 2.5 s. The buffers
  in transit are ordinary memory: with the machine at a load of 33 to 40 the
  copies to the devices took 0.9 to 1.5 s instead of 0.25 s, which
  page-locked buffers would not.
- **Across loads.** The state that comes back is bit for bit the state that
  was written, so a resume from SSD in the same load equals a resume from the
  devices where the same checkpoint is used. A later load may choose other
  kernel layouts for its products (they are measured at load), whose rounding
  differs from the load that wrote the state: the reply after a restart is
  the reply a server gives that continues that state, which is not bit for
  bit what the new load would have computed for the prefix itself. The texts
  of the tests, which copy from their prompts, are the same.

A randomised test drives sequences through appends, rollbacks, forks, cache
hits and evictions against a simulated device and checks after every
operation that each sequence reads exactly the state its own tokens produce,
that nothing is written into a shared page, and that no page leaks.
