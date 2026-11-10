//! The engine's own storage formats for the K and V rows of attention layers.
//!
//! The K and V rows of attention layers are stored in pages on the GPU, and
//! VRAM is the binding limit, so the format of a row decides how much context
//! fits. llama.cpp offers a handful of KV types under names borrowed from its
//! weight formats (`q4_0`, `q5_0`, `q8_0`), with nothing between 5 and 8 bits.
//! The engine has one ladder instead, with plain names, chosen separately for K
//! and for V:
//!
//! | name  | bytes per 32 values | bits per value | scales tried per group | mean per tile |
//! | ----- | ------------------: | -------------: | ---------------------: | ------------- |
//! | `f16` | 64                  | 16             | none                   | no            |
//! | `q8`  | 34                  | 8.5            | 1                      | no            |
//! | `q7`  | 30                  | 7.5 + 0.125    | 3                      | yes           |
//! | `q6`  | 26                  | 6.5 + 0.125    | 5                      | yes           |
//! | `q5`  | 22                  | 5.5 + 0.125    | 9                      | yes           |
//! | `q4`  | 18                  | 4.5 + 0.125    | 17                     | yes           |
//!
//! `q8` is the default for K and for V: against `f16` it loses hardly anything,
//! and it is the fast choice, so it and `f16` are the plain formats (a scale
//! from the largest magnitude of the group, nothing more). Below it the formats
//! work harder the narrower they get, and may be slower for it. Most of the work
//! is on the side of **writing**: a row is written once and read at every later
//! token, so a cost when writing is cheap and a cost when reading is not. The
//! narrower formats search for the scale of each group that minimises the squared
//! error instead of taking the largest magnitude, use every code of their width,
//! and let the sign of the scale decide on which side the extra code goes; none of
//! that changes what is stored or what a decoder does. They also store their rows
//! a **tile** of cells at a time, relative to the mean of the tile, which is
//! stored once per tile (0.125 bits per value for the tiles of 128 cells of
//! [`KvPolicy`]). That is the one technique with a cost when **reading**: one
//! extra term per tile and query, next to the thousands of terms of the tile's
//! rows (see "What attention does with each width"). Which other techniques were
//! measured, and why they are not in the formats, is in "Choosing the
//! techniques". The page-level policy that keeps the first and the newest
//! positions of a sequence in `f16` is [`KvPolicy`].
//!
//! This module is the specification of those formats and their CPU reference.
//! The attention kernels of both backends are held to it bit for bit: the
//! golden vectors in `tests/golden/kv/` pin its output, and everything below is
//! written so that a kernel can reproduce every byte and every float.
//!
//! # Specification
//!
//! ## Rows and groups
//!
//! A row is the K or V values of one head for one position. Its length is a
//! multiple of 32 (typically 128 or 256). A row is stored as consecutive groups
//! of 32 values, and each group is encoded on its own, so a group is the unit a
//! kernel works on. All multi-byte numbers are little endian.
//!
//! ## `f16`
//!
//! A group is its 32 values as IEEE binary16, 64 bytes. Each value is rounded to
//! nearest even. A finite value beyond the range of f16 does not become
//! infinity, which would poison every dot product it enters: it saturates to
//! +-65504, the largest finite f16 (the same rule the scale of the `qB` formats
//! follows).
//!
//! ## `qB`, for B = 8, 7, 6, 5 or 4
//!
//! Let `qmax = 2^(B-1) - 1`, which is 127, 63, 31, 15 and 7. A group is encoded
//! in four steps. The rotation (1), the rule that turns a largest magnitude into
//! a scale (2) and the layout (4) are the same for all five widths. Step 3, which
//! makes the scale and the codes, is a plain rule for `q8` and a search, built on
//! step 2, for the others.
//!
//! **1. Rotation.** The 32 values `x` are rotated with the 32-point
//! Walsh-Hadamard transform in natural (Sylvester) order, scaled by `1/sqrt(32)`
//! so that it is orthonormal: `r[k] = C * sum_j (-1)^popcount(j & k) * x[j]`.
//! The order of the additions is part of the definition, because float addition
//! is not associative and kernels must produce the same bits. It is the
//! butterfly network with the stage width `h` growing from 1 to 16:
//!
//! ```text
//! for h in 1, 2, 4, 8, 16:                      (in this order)
//!     for i in 0, 2h, 4h, ..., 32 - 2h:
//!         for j in i .. i + h:
//!             a = x[j];  b = x[j + h]           (the old values)
//!             x[j]     = a + b                  (the lower element gets the sum)
//!             x[j + h] = a - b                  (the upper one the difference)
//! for j in 0 .. 32:
//!     r[j] = x[j] * C                           (one f32 multiply, after all stages)
//! ```
//!
//! `C` is the `f32` nearest to `1/sqrt(32)`, bit pattern `0x3e3504f3`
//! (0.176776692...). Every addition, subtraction and multiplication is a plain
//! f32 operation with round to nearest even, and none may be fused with another
//! (there is no multiply followed by an add in the network, so a compiler has
//! nothing to fuse as long as the final multiply stays separate).
//!
//! The rotation is its own inverse, `H * H = I`: applying it to `r` gives back
//! `x`, up to float rounding. It spreads the energy of an outlier over all 32
//! values of the group, so the largest magnitude, which sets the step, is much
//! closer to the typical one and the step shrinks.
//!
//! **2. The scale of a largest magnitude.** For a largest magnitude `a`, above
//! zero, the scale `s(a)` is an f16 found from `q = a / qmax`, one f32 division:
//!
//! - `s(a)` is the **smallest positive f16 that is not below `q`**. It is rounded
//!   up, never to nearest. Equivalently: round `q` to the nearest f16 (ties to
//!   even); if the result is zero or below `q`, take the next f16 up (the bit
//!   pattern plus one).
//! - If that is more than 65504 (the largest finite f16), `s(a) = 65504`
//!   (`0x7bff`). This is the one case in which a code can fall outside the range
//!   of the largest magnitude; it needs `a` above `65504 * qmax`, far beyond real
//!   K/V.
//!
//! Rounding up is what keeps the largest value inside the code range, so that the
//! error of every value is at most half a step: the f16 nearest to `q` can be
//! below it, and where the f16 has few bits (a small scale in the subnormal range
//! has as little as one) a scale rounded down by up to half would let the codes
//! reach 1.5 times `qmax`. `s(a)` is never 0 (the smallest is the f16 subnormal
//! `2^-24`, `0x0001`), so the division in step 3 is always defined.
//!
//! **3. Scale and codes of `q8`.** Let `amax` be the largest `|r[j]|`. If
//! `amax == 0` (a group of zeros, `-0.0` included), the stored scale is `+0`
//! (`0x0000`) and all codes are 0. Otherwise the stored scale is `s(amax)` and
//! `c[j] = clamp(rint(r[j] / s), -qmax, qmax)`, with `s` the stored scale as f32.
//! The division is one correctly rounded f32 division (not a multiplication by a
//! reciprocal) and `rint` rounds halves to even. The clamp is symmetric: the code
//! `-2^(B-1)` is never written by `q8`, and its scale is never negative. Because
//! `s >= amax / qmax`, the clamp only acts when the scale saturated, and every
//! value is within half a step of its code.
//!
//! **3. Scale and codes of `q7`, `q6`, `q5` and `q4`: the search.** The scale is
//! chosen among candidates, and the codes and the stored scale are those of the
//! candidate with the least squared error. The codes may now be any B-bit
//! two's complement number, `-qmax - 1 ..= qmax`: the extra code `-2^(B-1)` is
//! used, on the side of the larger extreme of the group, by giving the scale the
//! sign that puts it there. Let `amax` be the largest `|r[j]|` (if it is 0 the
//! group is stored as for `q8`: scale `+0`, codes 0), `pmax` the largest `r[j]`
//! and `nmax` the largest `-r[j]`, each at least 0. The sign is `-1` if
//! `pmax > nmax` and `+1` otherwise (a tie is positive). The candidates are
//! tried for `j = 0, 1, ..., 2^(8-B)` in this order (`j` runs to 2, 4, 8 and 16
//! for `q7`, `q6`, `q5` and `q4`, so that 3, 5, 9 and 17 scales are tried):
//!
//! ```text
//! a   = amax * ((64 - j) / 64)       one f32 multiply; the factor is exact
//! s0  = s(a)                         the positive scale of step 2 for a
//! g   = sign * s0                    as f32; the scale the codes are made with
//! c[i]= clamp(rint(r[i] / g), -qmax - 1, qmax)       one f32 division each
//! n   = sum of c[i]^2                an integer, at most 32 * 128^2: exact
//! if n == 0:
//!     scale = s0                     all codes zero: the sign is left positive
//! else:
//!     p     = sum32(r[i] * c[i])     f32 products, then the sum below
//!     fit   = min(|p| / n, 65504)    one f32 division
//!     scale = max(nearest f16 of fit, 0x0001), with the sign bit set if sign = -1
//! d[i]= r[i] - c[i] * scale          the product is exact, the subtraction rounds
//! e   = sum32(d[i] * d[i])
//! keep this candidate if j == 0 or e < the least e so far     (a tie keeps the earlier)
//! ```
//!
//! `sum32` is the sum of 32 values with the pairs at distance 16 added first,
//! then those at 8, 4, 2 and 1: `for h in 16, 8, 4, 2, 1: for i in 0 .. h:
//! v[i] = v[i] + v[i + h]`, the result `v[0]`. That is the order in which a warp
//! of 32 lanes gets it from five exchanges. As in the rotation, every operation
//! is a plain f32 operation and none may be fused with another (a product
//! followed by a sum, above, is two operations); the sums above can overflow to
//! infinity for groups near `f32::MAX`, where all candidates tie at an infinite
//! error and the first, `j = 0`, is kept.
//!
//! The first candidate, `j = 0`, is exactly the scale and the codes `q8` would
//! make (the extra code is not reached and a negative scale only mirrors the
//! codes), and the least-squares fit and the choice among candidates can only
//! lower the squared error of the group, so **a group of `q7` to `q4` is never
//! stored with more squared error than the rule of `q8` gives it** (up to the
//! rounding of the f32 sums that compare the candidates). Values
//! beyond the range of a reduced scale are clipped on purpose, so the bound of
//! half a step per value holds for `q8` only.
//!
//! The fitted scale is the one for which the error `r - c * scale` is orthogonal
//! to the decoded values `c * scale`: the decoded group is the projection of the
//! group on its codes. That, more than the choice of the clip ratio, is what
//! helps the dot products of attention: the error is uncorrelated with the group.
//! It also lets a group of nearly equal magnitudes (a single large value
//! rotated, plus small values) be stored with the error of its small values
//! alone, where a scale from the largest magnitude loses about five times that.
//!
//! **4. Layout.** A group is the scale (2 bytes) followed by the 32 codes as a
//! bit stream of `4 * B` bytes: code `i` occupies the stream bits
//! `[i * B, (i + 1) * B)`, where stream bit `n` is bit `n % 8` of byte `n / 8`
//! (the least significant bit comes first), each as a B-bit two's complement
//! number. So a group is `2 + 4 * B` bytes, `B + 0.5` bits per value: 34 bytes for
//! `q8`, 30 for `q7`, 26 for `q6`, 22 for `q5` and 18 for `q4`. For `q4` that is
//! the low nibble of byte `2 + j` holding code `2j` and the high nibble code
//! `2j + 1`. The scale is an f16 pattern like any other: for `q7` to `q4` its
//! sign bit may be set.
//!
//! ## Tiles and their mean (`q7` to `q4`)
//!
//! The K rows of a head at neighbouring positions have much in common, and so do
//! its V rows: on real models a quarter to a third of their energy is the mean of
//! each value over the positions. `q7` to `q4` do not spend codes on it. Their
//! rows are stored a **tile** at a time: a tile is the rows of one head (its K
//! rows, or its V rows) for consecutive cells, `T` of them, at least one. The
//! tile stores the mean of its rotated rows once and every row relative to it.
//! With `r_i` the rotated row of cell `i` of the tile (step 1 above, group by
//! group) and `n` the length of a row:
//!
//! ```text
//! for j in 0 .. n:                             (every rotated position of the row)
//!     sum = +0
//!     for i in 0 .. T:  sum = sum + r_i[j]     (f32, in the order of the cells)
//!     m       = sum / T                        (one f32 division; T as f32)
//!     mean[j] = the f16 nearest to clamp(m, -65504, 65504)        (ties to even)
//! for i in 0 .. T, j in 0 .. n:
//!     d_i[j] = r_i[j] - mean[j]                (the stored mean as f32; one subtraction)
//! ```
//!
//! Each row is then encoded by steps 3 and 4 with `d_i` in the place of `r`,
//! group by group as before. The mean is stored as `n` f16, little endian, in the
//! order of the rotated positions: [`KvFormat::mean_bytes`] is `2 n`. Where the
//! mean lives next to the rows is the business of whoever owns the pages; for
//! tiles of `T` cells it adds `16 / T` bits per value, 0.125 for the 128 cells of
//! [`TILE_CELLS`].
//!
//! The order of the additions is part of the definition, as in the rotation. The
//! encoder subtracts the mean as it is stored, so the rounding of the mean to f16
//! costs nothing: whatever it leaves is part of `d_i` and goes into the codes. A
//! sum that overflows f32 is an infinity, which the clamp turns into `+-65504`;
//! the sum cannot be NaN, because all `r_i[j]` are finite. A row encoded on its
//! own with [`encode_row`] is the same as a tile whose stored mean is `+0`
//! everywhere.
//!
//! `f16` and `q8` have no mean: a tile of them is its rows, each as above.
//!
//! ## Decoding
//!
//! Decoding a `qB` group gives the **rotated** values `c[j] * s`. The product
//! is exact in f32 (a code has at most 8 bits and an f16 scale at most 11), so
//! this step has exactly one correct result and there is no rounding for a
//! kernel to get wrong, fused multiply-add included. A decoder reads a stored
//! code as plain two's complement, so the pattern `-2^(B-1)`, which `q8` never
//! writes and the narrower formats do, decodes to `-2^(B-1) * s`; the scale is
//! used as it stands, with its sign, a NaN or infinite one included. Un-rotating,
//! which is the same transform again, is a separate step: [`decode_row`] is
//! [`decode_row_rotated`] followed by [`rotate_groups`].
//!
//! A row of a tile of `q7` to `q4` stands for `c[j] * s + mean[j]`, one f32
//! addition after the exact product, with the mean of its tile widened from
//! f16. [`decode_tile_rotated`] gives that for all rows of a tile and
//! [`decode_tile`] un-rotates it. [`decode_row_rotated`] on a row of a tile
//! still gives `c[j] * s`, which is what the kernels read; the mean enters
//! attention once per tile, as described below.
//!
//! The split is the point of the design. A rotation does not change a dot
//! product, so attention rotates the query once per group and takes its dot
//! product with the rotated keys exactly as stored, never un-rotating K; and it
//! accumulates `softmax * V` in rotated space and un-rotates the finished
//! result once per group instead of once per cached position. The `f16` format
//! is not rotated, and for it both decoders return the stored values
//! ([`KvFormat::is_rotated`] says which a format is); K and V are chosen
//! separately, so the query is rotated only when K is a `qB` format and the
//! attention output is un-rotated only when V is one.
//!
//! ## What attention does with each width
//!
//! This is what a kernel must do, per head and position, and it is the same for
//! `q8` to `q4` except for how the codes are unpacked:
//!
//! - **K.** Rotate the query once per head and token with the 32-point transform
//!   above, group by group (not over the whole head). For each cached position
//!   and each group, `dot = scale * sum_j (query_rotated[j] * code[j])`, summed
//!   over the groups in any order that suits the kernel. That is one multiply by
//!   the f16 scale per group and position, in whatever sign it has, on top of the
//!   products with the codes. No further term per group, no offset, no table
//!   lookup, no un-rotation of K.
//! - **V.** Accumulate `softmax_weight * scale * code[j]` into the rotated
//!   output, one scale per group and position again, and un-rotate the finished
//!   output once per head and group.
//! - **`f16`**: no rotation, no scale; plain halves.
//! - **`q8`**: the codes are one signed byte each; nothing to unpack.
//! - **`q7`, `q6`, `q5`**: the codes are a bit stream of 7, 6 or 5 bits each;
//!   unpacking is extra work that grows with the narrowness of the stream.
//! - **`q4`**: two codes per byte, the low nibble first.
//!
//! For `q7` to `q4` the mean of each tile adds one term per tile, not per
//! position:
//!
//! - **K.** Once per tile and query, `mean_dot = sum_j (query_rotated[j] *
//!   mean[j])` over the whole row, and `mean_dot` is added to the `dot` of every
//!   position of that tile before the softmax. For a row of 256 values and tiles
//!   of 128 cells that is 256 products next to the 32 768 of the tile's rows.
//! - **V.** Per tile, sum the softmax weights of its positions, and add
//!   `weight_sum * mean[j]` to the rotated output, before it is un-rotated.
//!
//! Both follow from a row standing for `code * scale + mean` and are exact
//! rearrangements of it; as float arithmetic they differ from
//! [`decode_tile_rotated`] by rounding only. Positions kept in `f16`
//! ([`KvPolicy`]) have no mean.
//!
//! Writing a row costs more the narrower the format: [`KvFormat::scale_candidates`]
//! scales are tried per group (1 for `q8`, then 3, 5, 9 and 17), each with 32
//! divisions, products and a sum of 32. The mean of a tile costs one addition per
//! value when the tile is written, and one subtraction.
//!
//! # Choosing the techniques
//!
//! The rule for what may be added to a format is that a row is written once and
//! read at every later token. The candidates were ordered by what they cost at
//! read time and adopted per width only where they measurably help:
//!
//! 1. **write-only** (the stored format and the decoder do not change): the
//!    search of the scale by squared error, with the extra code and the sign of
//!    the scale, and the least-squares fit of the scale. In the formats.
//! 2. **one extra term per group at read time**: an offset per group (a group
//!    mean, or an asymmetric grid from the midrange), which attention handles with
//!    the sum of the query per group. Not in the formats.
//! 3. **a different rotation**: one transform over blocks of 64, 128 or 256
//!    values, or the whole row, instead of per group of 32. It costs the same at
//!    read time. Not in the formats.
//! 4. **page-level policy** outside the codec of a row: [`KvPolicy`].
//! 5. **one extra term per tile at read time**: the mean of every rotated value
//!    over the cells of a tile, stored once per tile. In the formats `q7` to `q4`.
//!    It was not among the first candidates; it came out of the measurement on
//!    real rows (below).
//!
//! Anything that needs table lookups or an un-rotation per value at read time was
//! out from the start.
//!
//! **The choices were made on synthetic rows and then checked on real ones.** The
//! numbers in this section are from synthetic rows: Gaussian with outliers, a few
//! loud channels, heavy tails (Student's t with three degrees of freedom), fixed
//! per-channel offsets, a small non-zero mean, and, as stress cases, a single
//! large value among small ones and a large mean. The first five were taken as
//! realistic, the last two as stress. Afterwards the formats were measured on the
//! K and V rows of two real models (`docs/reference/kv-real-data.md`, one text of
//! 3.7k tokens, error of attention, not of logits): the gain of the ladder over
//! the rule of `q8` is the same there (0.8, 1.0 and 1.6 dB at q6, q5 and q4, in
//! every layer), real rows are less peaky than the synthetic outlier rows (the
//! rotation is worth about 2 dB, not 7 to 9), and the stress case of a large mean
//! does not occur. What real rows have, and the synthetic kinds did not show, is a
//! mean per channel over neighbouring positions that holds a quarter to a third
//! of the energy. That is the mean of a tile of `q7` to `q4`: stored once per 128
//! cells, for 0.125 bits per value, it lowers the error of the attention output
//! by 25%, 27% and 32% at q6, q5 and q4 on the smaller model (19%, 21% and 25% on
//! the larger) and the KL divergence of the attention weights by 42% to 49%,
//! measured with the codec as built and the cells that [`KvPolicy`] keeps exact;
//! no layer got worse. kvarn's normalisation of a tile along both axes, measured
//! the same way on top of these formats, changed nothing (the scale per group
//! already follows the size of each row). The test `kv_techniques` prints the
//! complete tables (`cargo test -p ares-quant --test kv_techniques --
//! --nocapture`): for every kind of row and width the signal-to-noise ratio, the
//! error of dot products `q.k` (with queries of the same kind as the keys), the
//! same with the part common to all keys removed (a softmax does not see it), and
//! the error of a weighted sum of rows (what attention takes of V). Its candidates
//! are in one table, and in the source the choice of which widths search is in
//! `encode_group` and how many scales they try in `search_steps`, so that
//! revisiting is a matter of changing a line.
//!
//! Gain over the rule of `q8` (a scale from the largest magnitude), in dB, mean
//! over the five realistic kinds, rows of 256 values (rows of 128 give the same
//! within 0.2 dB):
//!
//! | what is added                          | q7  | q6  | q5  | q4  | measure |
//! | -------------------------------------- | --: | --: | --: | --: | ------- |
//! | the search by squares alone            | 0.0 | 0.2 | 0.4 | 0.6 | SNR     |
//! |                                        | 0.0 | 0.1 | 0.2 | 0.3 | q.k     |
//! | + the least-squares fit of the scale   | 0.2 | 0.3 | 0.4 | 0.6 | SNR     |
//! |                                        | 1.2 | 1.1 | 0.9 | 0.6 | q.k     |
//! | + the extra code (positive scale)      | 0.4 | 0.6 | 0.8 | 1.1 | SNR     |
//! |                                        | 1.4 | 1.3 | 1.3 | 1.1 | q.k     |
//! | + the sign of the scale (**the ladder**) | 0.6 | 0.8 | 1.1 | 1.6 | SNR     |
//! |                                        | 1.6 | 1.5 | 1.7 | 1.6 | q.k     |
//! |                                        | 1.6 | 1.7 | 1.9 | 2.4 | q.k, common part removed |
//! |                                        | 0.7 | 0.8 | 1.1 | 1.5 | weighted sum of rows |
//!
//! Every step raises the signal-to-noise ratio or the dot products by 0.2 dB or
//! more at every width (the fit mostly the dot products, by 0.3 to 1.2 dB; the
//! extra code and the sign of the scale both, by 0.2 to 0.5 dB each), and none of
//! them loses signal-to-noise ratio on any of the seven kinds. The fit is what
//! turns the dot products from nearly unchanged into 1 dB better. The search by
//! squares alone *loses* up to 2 dB of `q.k` on one realistic kind at q4 (fixed
//! channel offsets, where the error that clipping puts on the loud channels is the
//! same in every key); the part common to all keys does not reach a softmax, and
//! with it removed that kind gains 2.8 dB. The gains grow toward the narrow
//! widths, as the effort does: the search tries `2^(8-B)` reduced scales, up to
//! the ratio `1 - 2^(8-B)/64`, twice as far as the ratio `qmax / (qmax + 1)` at
//! which the extra code pays. (In exploration with a positive scale only, going
//! from 8 to 16 reduced scales at q4 took 0.1 dB more and 32 nothing; at q5 4 to 8
//! took 0.1 dB.)
//!
//! The one stress case where the ladder loses is a large mean in every group,
//! where the rotation turns the mean into one dominant value, which the search
//! then clips a little: `q.k` is 1.0, 3.4, 6.1 and 9.6 dB worse at q7 to q4 on
//! rows whose dot products are dominated by that mean (a pairing of keys and
//! queries that real rows are not known to have; SNR and weighted sums gain there
//! too). The real K and V rows measured so far do not have it: no group of 32 has
//! even half of its energy in its mean, and the ladder beats the rule of `q8` in
//! every layer. Should a model show it, fewer scales tried is the knob
//! (`search_steps`), though the fit alone was no safer in exploration.
//!
//! Not adopted, with the same measure (mean over the realistic kinds, SNR at q4
//! unless stated; "+bits" is what it costs in a row of 256 values):
//!
//! | candidate                              | +bits | gain (SNR at q4)               | why not |
//! | --------------------------------------- | ----: | ------------------------------ | ------- |
//! | rotation over blocks of 64              | 0     | -0.5 dB against the ladder     | worse |
//! | rotation over blocks of 128, 256        | 0     | -0.9, -1.1 dB against the ladder | worse |
//! | the same with the scale of `q8`         | 0     | -1.0 dB; one large value 34.5 dB against 43.0 at q4 (and 34.6 against 42.9 at q5) | worse: the weakness of a single large value is eight times larger |
//! | offset: the mean of each group          | 0.5   | +0.4 dB alone, +0.4 with the ladder | half a bit is worth 3 dB; only a large mean in every group gains (11 dB) |
//! | offset: the mean of the row             | 0.06  | +0.2 dB; 3.2 dB worse on one large value | helps a mean shared by all groups, hurts a row that has it in one |
//! | offset: the midrange (asymmetric grid)  | 0.5   | +1.1 dB alone, +0.1 dB with the ladder | the ladder takes most of it for free |
//!
//! These offsets are means over the values of one row. The mean of a tile is over
//! positions, per value, and is in the formats.
//!
//! The reason for the wider rotation losing is that the rotation of a group of 32
//! keeps an outlier's energy in its own group, where its group's scale follows it
//! and the other groups keep a fine step; a rotation over 256 values spreads the
//! outlier to all eight groups and every group's step grows. A single large value,
//! rotated over the whole row, is a pattern of equal magnitudes in all eight groups
//! instead of one, so its weakness is eight times as large.
//!
//! ## Measured quality
//!
//! Signal-to-noise ratio of encoding and decoding 64 rows of 256 values (the
//! test `signal_to_noise_per_width_with_and_without_the_techniques_and_the_rotation`
//! prints them), in dB. "Built" is the formats; "before" the same rotated codec
//! with the rule of `q8` at every width (a scale from the largest magnitude, which
//! is what `q7` to `q4` were before the search; it is the same for `q8` and
//! `f16`); "not rotated" is that rule applied to the unrotated group:
//!
//! | data                                       |   f16 |    q8 |    q7 |    q6 |    q5 |    q4 |
//! | ------------------------------------------ | ----: | ----: | ----: | ----: | ----: | ----: |
//! | Gaussian with outliers, built              |  73.8 |  47.3 |  41.9 |  35.8 |  29.9 |  23.7 |
//! | Gaussian with outliers, before             |  73.8 |  47.3 |  41.3 |  35.0 |  28.7 |  22.1 |
//! | Gaussian with outliers, not rotated        |  73.8 |  39.8 |  33.9 |  27.6 |  21.4 |  14.8 |
//! | fixed outlier channels, built              |  73.6 |  47.6 |  42.2 |  36.2 |  30.1 |  24.0 |
//! | fixed outlier channels, before             |  73.6 |  47.6 |  41.5 |  35.3 |  29.0 |  22.4 |
//! | fixed outlier channels, not rotated        |  73.6 |  39.7 |  33.6 |  27.5 |  21.1 |  14.8 |
//! | one large value, rest small, built         | 115.5 |  53.0 |  46.1 |  50.9 |  50.9 |  50.7 |
//! | one large value, rest small, before        | 115.5 |  53.0 |  45.1 |  43.0 |  43.0 |  43.0 |
//! | one large value, rest small, not rotated   | 115.5 |  50.9 |  50.9 |  50.9 |  50.9 |  50.6 |
//!
//! Each bit is worth about 6 dB. The rotation is worth 7 to 9 dB at every width
//! when rows have outliers, because it brings the largest magnitude of a group,
//! which sets the step, down towards the typical one. It was not a win everywhere:
//! when one value is a couple of thousand times the rest, the rotation turns its
//! group into 32 values of nearly the same size, which a scale from the largest
//! magnitude quantizes all to the top code, so that they decode as that maximum and
//! the error is the spread of the small values around it, about five times their
//! own power (the "before" rows), where the unrotated quantizer simply drops them
//! (an error equal to their power). The fitted scale puts the common magnitude
//! where the values are, and the error becomes their own power, as unrotated
//! ("built" at q6 and below). At q7 the small values are about a fifth of a step
//! and the group of nearly equal values straddles the boundary between two codes;
//! a reduced scale would put all of them on the top code, but with the extra code
//! on one side only that is 63 on one side and 64 on the other, and no single
//! scale fits both (without the extra code q7 gets 50.9 dB on this kind, and 0.4
//! dB less on the realistic ones). So q7 is lower than q6 on this one kind of row:
//! the error does not shrink monotonically with the width here, as it does on
//! every realistic kind.
//!
//! ## Exact cells
//!
//! Of a sequence, the first and the newest cells may be kept in `f16` instead
//! (kvarn does, and its largest gain at 4 bits is there). That is a policy for the
//! owner of the pages, not part of the codec; [`KvPolicy`] holds the numbers
//! (defaults per width, which cells are stored in the formats, and the bytes a
//! sequence takes under them). It is also what makes a tile simple to write: the
//! cells of the newest tiles are in `f16` anyway, and a tile is converted when it
//! is full and leaves that window, with all of its rows at hand for the mean.
//!
//! ## What is defined at the edges
//!
//! - Every input value must be finite: [`encode_row`] returns an error naming
//!   the first NaN or infinity, for every format.
//! - A group so large that the rotation itself overflows f32 (sums of values near
//!   `f32::MAX`) is an error naming the group. Anything below that encodes.
//! - A group with a value that makes `amax / qmax` exceed 65504 saturates the
//!   scale as above (to `+-65504`, the sign by the rule of its format); for `f16`
//!   the value itself saturates. A saturated group decodes to finite values
//!   nowhere near the input: `+-qmax * 65504` for `q8`, and for `q7` to `q4`
//!   `+-(qmax + 1) * 65504` on the side of the extreme.
//! - The sign of a zero is not stored by the `qB` formats; `f16` keeps it. A group
//!   whose codes are all zero (zeros, or values so small the scale is the
//!   smallest f16) has a positive scale in every format.
//! - A row whose length is not a multiple of 32, or a byte slice of the wrong
//!   size, is an error, and no function writes anything when it returns one.
//! - A tile has at least one row, and its values are whole rows; its mean slice
//!   is exactly [`KvFormat::mean_bytes`] long, which is empty for `f16` and `q8`.
//!   A tile of one row stores that row, rounded to f16, as its mean and what the
//!   rounding left in the codes.
//! - A mean beyond the range of f16 saturates to `+-65504` (a sum that overflowed
//!   f32 included), and the rows are stored relative to the saturated mean, so
//!   they then saturate their scales as above. A mean of zero has the sign of the
//!   quotient `sum / T`: `+0` unless the sum is negative and the quotient
//!   underflows; the pattern stored is whatever the f16 conversion gives.
//!
//! # Example
//!
//! The attention side of a K row, with the query rotated instead of the keys:
//!
//! ```
//! use ares_quant::{KvFormat, decode_row_rotated, encode_row, rotate_groups};
//!
//! let format: KvFormat = "q5".parse().unwrap();
//! let key: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
//! let mut query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.11).cos()).collect();
//!
//! let mut stored = vec![0; format.row_bytes(key.len()).unwrap()];
//! encode_row(format, &key, &mut stored).unwrap();
//! assert_eq!(stored.len(), 4 * 22); // four groups of 22 bytes
//!
//! let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
//! let exact: f32 = key.iter().zip(&query).map(|(k, q)| k * q).sum();
//! let size = norm(&key) * norm(&query);
//!
//! rotate_groups(&mut query).unwrap();
//! let mut rotated_key = vec![0.0; key.len()];
//! decode_row_rotated(format, &stored, &mut rotated_key).unwrap();
//! let approx: f32 = rotated_key.iter().zip(&query).map(|(k, q)| k * q).sum();
//! // q5 costs about 1% of |key| |query| here, a dot product of unrelated rows.
//! assert!((exact - approx).abs() < 0.02 * size, "{exact} vs {approx}");
//! ```
//!
//! The same for the K rows of a tile, whose mean enters once for all of them:
//!
//! ```
//! use ares_quant::{KvFormat, decode_row_rotated, encode_tile, rotate_groups};
//!
//! let format = KvFormat::Q4;
//! let (n, cells) = (64, 128);
//! // Keys that share an offset per value, as the rows of neighbouring positions do.
//! let keys: Vec<f32> = (0..cells * n)
//!     .map(|i| ((i % n) as f32 * 0.9).sin() + 0.3 * (i as f32 * 0.37).sin())
//!     .collect();
//! let mut query: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos()).collect();
//!
//! let row_bytes = format.row_bytes(n).unwrap();
//! let mut mean = vec![0; format.mean_bytes(n).unwrap()];
//! let mut stored = vec![0; cells * row_bytes];
//! encode_tile(format, n, &keys, &mut mean, &mut stored).unwrap();
//! assert_eq!(mean.len(), 2 * n); // one f16 per value, once for the 128 rows
//!
//! // The mean is laid out like a row of f16 in the rotated domain.
//! let mut mean_rotated = vec![0.0; n];
//! decode_row_rotated(KvFormat::F16, &mean, &mut mean_rotated).unwrap();
//!
//! let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
//! let exact: Vec<f32> = keys.chunks(n).map(|key| dot(key, &query)).collect();
//!
//! rotate_groups(&mut query).unwrap();
//! let mean_dot = dot(&mean_rotated, &query); // once per tile and query
//! let mut row = vec![0.0; n];
//! for (bytes, exact) in stored.chunks(row_bytes).zip(exact) {
//!     decode_row_rotated(format, bytes, &mut row).unwrap();
//!     let approx = dot(&row, &query) + mean_dot;
//!     // The codes only carry what differs from row to row, the part of size 0.3.
//!     assert!((exact - approx).abs() < 0.3, "{exact} vs {approx}");
//! }
//! ```

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use crate::error::Error;
use crate::half::{f16_to_f32, f32_to_f16};

/// Values per group, the unit of rotation, scale and packing.
pub const GROUP_VALUES: usize = 32;

/// Cells per tile under a [`KvPolicy`]: the consecutive cells of a sequence
/// whose rows of `q7` to `q4` share one stored mean. Tiles are aligned to the
/// start of the sequence: tile `t` is the cells `128 t .. 128 (t + 1)`.
///
/// The codec itself ([`encode_tile`]) takes any number of rows; this is the
/// number the policy and its sizes use. Measured on real K and V at `q4`
/// before the mean was built (`docs/reference/kv-real-data.md`, section 6),
/// tiles of 32, 128 and 512 cells lowered the error of the attention output by
/// 36%, 31% and 28%, for 0.5, 0.125 and 0.03 bits per value.
pub const TILE_CELLS: usize = 128;

/// The largest finite f16, where scales and `f16` values saturate.
const F16_MAX: f32 = 65504.0;
/// The bit pattern of [`F16_MAX`].
const F16_MAX_BITS: u16 = 0x7bff;
/// The first f16 pattern that is not finite (infinity).
const F16_INFINITY_BITS: u16 = 0x7c00;

/// `1/sqrt(32)` as the nearest `f32`: the factor that makes the 32-point
/// Hadamard transform orthonormal. It is given as bits so that a kernel can
/// take the identical constant.
const INV_SQRT_32: f32 = f32::from_bits(0x3e35_04f3);

/// A storage format for K or V rows. See the [module documentation](self) for
/// the specification of each.
///
/// The quantized formats are separate variants, one per width, so that a width
/// outside 4 to 8 cannot be written down.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvFormat {
    /// 32 IEEE half floats per group.
    F16,
    /// 8-bit codes: 34 bytes per group.
    Q8,
    /// 7-bit codes: 30 bytes per group.
    Q7,
    /// 6-bit codes: 26 bytes per group.
    Q6,
    /// 5-bit codes: 22 bytes per group.
    Q5,
    /// 4-bit codes: 18 bytes per group.
    Q4,
}

impl KvFormat {
    /// Every format, from the largest to the smallest.
    pub const ALL: [KvFormat; 6] =
        [KvFormat::F16, KvFormat::Q8, KvFormat::Q7, KvFormat::Q6, KvFormat::Q5, KvFormat::Q4];

    /// The name used in configuration and messages: `f16`, `q8`, ... `q4`.
    pub const fn name(self) -> &'static str {
        match self {
            KvFormat::F16 => "f16",
            KvFormat::Q8 => "q8",
            KvFormat::Q7 => "q7",
            KvFormat::Q6 => "q6",
            KvFormat::Q5 => "q5",
            KvFormat::Q4 => "q4",
        }
    }

    /// The quantized format with `bits`-bit codes, or `None` outside 4 to 8.
    pub const fn quantized(bits: u32) -> Option<KvFormat> {
        match bits {
            8 => Some(KvFormat::Q8),
            7 => Some(KvFormat::Q7),
            6 => Some(KvFormat::Q6),
            5 => Some(KvFormat::Q5),
            4 => Some(KvFormat::Q4),
            _ => None,
        }
    }

    /// The width `B` of the codes of a `qB` format, or `None` for `f16`.
    pub const fn quant_bits(self) -> Option<u32> {
        match self {
            KvFormat::F16 => None,
            KvFormat::Q8 => Some(8),
            KvFormat::Q7 => Some(7),
            KvFormat::Q6 => Some(6),
            KvFormat::Q5 => Some(5),
            KvFormat::Q4 => Some(4),
        }
    }

    /// Whether the stored values are in the rotated domain, so that attention
    /// must rotate the query (for K) and un-rotate the result (for V). True for
    /// every `qB` format and false for `f16`.
    pub const fn is_rotated(self) -> bool {
        self.quant_bits().is_some()
    }

    /// Bytes of one group of 32 values: 64 for `f16`, `2 + 4 * B` for `qB`.
    pub const fn group_bytes(self) -> usize {
        match self.quant_bits() {
            None => 2 * GROUP_VALUES,
            Some(bits) => 2 + GROUP_VALUES / 8 * bits as usize,
        }
    }

    /// Bits per value of a row, including the scale: 16 for `f16`, `B + 0.5` for
    /// `qB`. Exact in `f32`. The mean of a tile of `q7` to `q4` comes on top:
    /// 16 bits per value of a row once per tile, 0.125 per value for tiles of
    /// [`TILE_CELLS`] cells.
    pub const fn bits_per_value(self) -> f32 {
        (self.group_bytes() * 8) as f32 / GROUP_VALUES as f32
    }

    /// How many scales the encoder tries for each group: none for `f16`, which
    /// has no scale, one for `q8`, and `2^(8-B) + 1` for `q7` to `q4`, that is
    /// 3, 5, 9 and 17. This is the cost of writing a row, which grows with the
    /// narrowness of the format; reading a row costs the same for all of them.
    pub const fn scale_candidates(self) -> u32 {
        match self.quant_bits() {
            None => 0,
            Some(8) => 1,
            Some(bits) => search_steps(bits) + 1,
        }
    }

    /// Whether the rows of this format are stored a tile at a time, relative to
    /// the mean of the tile (see "Tiles and their mean" in the
    /// [module documentation](self#tiles-and-their-mean-q7-to-q4)): true for `q7`
    /// to `q4`, false for `f16` and `q8`.
    pub const fn has_mean(self) -> bool {
        matches!(self, KvFormat::Q7 | KvFormat::Q6 | KvFormat::Q5 | KvFormat::Q4)
    }

    /// Bytes of the mean of a tile whose rows have `n_values` values: one f16
    /// per value for `q7` to `q4`, nothing for `f16` and `q8`.
    ///
    /// Fails when `n_values` is not a whole number of groups; a size that
    /// overflows `usize` is `usize::MAX`, as in [`KvFormat::row_bytes`].
    pub fn mean_bytes(self, n_values: usize) -> Result<usize, Error> {
        if !n_values.is_multiple_of(GROUP_VALUES) {
            return Err(Error::KvPartialGroup { values: n_values });
        }
        Ok(if self.has_mean() { n_values.saturating_mul(2) } else { 0 })
    }

    /// How many of the first cells of a sequence a [`KvPolicy`] keeps in `f16`
    /// by default when this is the format of K or V: none for `f16` and `q8`,
    /// the first tile ([`TILE_CELLS`], 128) for `q7` to `q4`.
    pub const fn default_exact_first_cells(self) -> usize {
        match self {
            KvFormat::F16 | KvFormat::Q8 => 0,
            KvFormat::Q7 | KvFormat::Q6 | KvFormat::Q5 | KvFormat::Q4 => TILE_CELLS,
        }
    }

    /// How many of the newest cells of a sequence a [`KvPolicy`] keeps in `f16`
    /// at least, by default, when this is the format of K or V: 0 for `f16` and
    /// `q8`, then 32, 64, 128 and 256 for `q7` to `q4`.
    pub const fn default_exact_newest_cells(self) -> usize {
        match self {
            KvFormat::F16 | KvFormat::Q8 => 0,
            KvFormat::Q7 => 32,
            KvFormat::Q6 => 64,
            KvFormat::Q5 => 128,
            KvFormat::Q4 => 256,
        }
    }

    /// Bytes of a row of `n_values` values.
    ///
    /// Fails when `n_values` is not a whole number of groups. A row so long that
    /// its size overflows `usize` gets `usize::MAX`, which no slice or ledger can
    /// hold, instead of wrapping.
    pub fn row_bytes(self, n_values: usize) -> Result<usize, Error> {
        if !n_values.is_multiple_of(GROUP_VALUES) {
            return Err(Error::KvPartialGroup { values: n_values });
        }
        Ok((n_values / GROUP_VALUES).saturating_mul(self.group_bytes()))
    }
}

impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for KvFormat {
    type Err = Error;

    /// Parses exactly the names [`KvFormat::name`] gives. Nothing else is
    /// accepted: not another case (`Q8`), not a llama.cpp name (`q4_0`, `q8_0`),
    /// not a width outside the ladder (`q3`, `q9`).
    fn from_str(name: &str) -> Result<Self, Error> {
        KvFormat::ALL
            .into_iter()
            .find(|format| format.name() == name)
            .ok_or_else(|| Error::KvFormatName { name: name.to_owned() })
    }
}

/// The valid names, comma separated, for the message of a rejected name.
#[derive(Debug)]
pub(crate) struct FormatNames;

impl fmt::Display for FormatNames {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, format) in KvFormat::ALL.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str(format.name())?;
        }
        Ok(())
    }
}

/// Bytes one cell (one position) takes in one attention layer: the K row and
/// the V row of every KV head, `kv_heads * (K row + V row)`, with K stored as
/// `k` and V as `v`.
///
/// `head_dim` must be a multiple of 32. With a size that overflows `usize` the
/// result is `usize::MAX`, as in [`KvFormat::row_bytes`].
pub fn kv_bytes_per_cell(
    k: KvFormat,
    v: KvFormat,
    kv_heads: usize,
    head_dim: usize,
) -> Result<usize, Error> {
    let per_head = k.row_bytes(head_dim)?.saturating_add(v.row_bytes(head_dim)?);
    Ok(per_head.saturating_mul(kv_heads))
}

/// Bytes the means of one tile take in one attention layer, on top of its
/// cells: for every KV head the mean of its K rows and the mean of its V rows,
/// each where the format has one ([`KvFormat::mean_bytes`]). Nothing when K and
/// V are `f16` or `q8`.
///
/// `head_dim` must be a multiple of 32, and an overflowing size is `usize::MAX`,
/// as in [`kv_bytes_per_cell`].
pub fn kv_mean_bytes_per_tile(
    k: KvFormat,
    v: KvFormat,
    kv_heads: usize,
    head_dim: usize,
) -> Result<usize, Error> {
    let per_head = k.mean_bytes(head_dim)?.saturating_add(v.mean_bytes(head_dim)?);
    Ok(per_head.saturating_mul(kv_heads))
}

const fn larger(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// How a sequence is stored: the formats of K and of V, and how many of its
/// first and its newest cells stay in `f16` instead.
///
/// This is page-level policy, outside the codec. A cell is one position of a
/// sequence. The first `exact_first_cells` and the last `exact_newest_cells` of
/// the `n` cells of a sequence keep both their K and their V rows in `f16`. The
/// other cells are stored in the formats `k` and `v`
/// ([`KvPolicy::quantized_cells`]):
///
/// - with `f16` and `q8` only, cell by cell: every cell between the two windows;
/// - when K or V is `q7` to `q4`, whose rows share a mean per tile, a tile of
///   [`TILE_CELLS`] cells at a time: a tile is stored in the formats once it is
///   full and none of its cells is in one of the two windows, and until then all
///   of its cells stay in `f16`. The window of the newest cells is therefore
///   `exact_newest_cells` to `exact_newest_cells + 127` cells wide, and a tile is
///   converted when it leaves that window, which is also when its mean is known.
///
/// When the two windows meet, every cell is exact. As a sequence grows, the cells
/// or tiles that leave the newest window are the ones to convert; whoever owns the
/// pages does that. This type holds the numbers and the arithmetic only.
///
/// A page should hold whole tiles. The stored rows of a tile depend on all of its
/// cells (through the mean), so a page that holds part of a tile is not a function
/// of the tokens up to its own end, which is what pages shared between sequences
/// are found by.
///
/// # Why exact cells, and how many
///
/// kvarn (beellama.cpp) keeps the first 128 and the newest 128 to 255 positions
/// of a sequence in `f16`, and that is where most of its gain at 4 bits comes
/// from (`docs/reference/kvarn-notes.md`). In the table of its author (Qwen 3.6
/// 27B, 64k context, median KLD against `bf16` KV) an `f16` tail of the newest
/// 128 tokens lowers the KLD of the plain 4-bit type by 35%, of the 5-bit one by
/// 16%, of the 6-bit one by 5% and of the 8-bit one by 1%; a tail of 1024 by 43%,
/// 19%, 5% and 1%. A user's measurement in its issue tracker (#109), which the
/// author agrees with, gives the exact tokens 35% of the gain at 4 bits and the
/// quantizer 8%. These are numbers of one model that the author measured; none
/// has been reproduced here. They say that the value of exact cells falls by a
/// factor of two to three with each bit of width, and that most of it is in the
/// first hundred or two cells (the first positions take much of the attention,
/// the newest ones are those the next token attends to most): the next 900 cells
/// buy 8 points at 4 bits and nothing at 6.
///
/// The cost is small. For 2 KV heads of 256 values an exact cell takes 2048 bytes
/// in one attention layer where a `q4` cell takes 576 (and the means of a tile
/// another 2048 per 128 cells). The 384 to 511 exact cells of the `q4` default
/// take 0.75 to 1.0 MiB instead of 0.21 to 0.28 MiB per layer and sequence, while
/// 64k cells of `q4` take 37 MiB.
///
/// So the defaults ([`KvFormat::default_exact_first_cells`] and
/// [`KvFormat::default_exact_newest_cells`]) are zero for `f16` and `q8` (the
/// latter is the default format and about speed, and 1% of a KLD that is already
/// close to `f16` is not worth a second kind of page). For `q7` to `q4` the first
/// tile stays exact, and the least width of the newest window halves for each
/// bit of width from 256 at `q4` (kvarn has 128 first and 128 to 255 newest):
///
/// | format | first cells | newest cells, at least |
/// | ------ | ----------: | ---------------------: |
/// | `f16`  | 0           | 0                      |
/// | `q8`   | 0           | 0                      |
/// | `q7`   | 128         | 32                     |
/// | `q6`   | 128         | 64                     |
/// | `q5`   | 128         | 128                    |
/// | `q4`   | 128         | 256                    |
///
/// The cells of a sequence are shared by K and V, so [`KvPolicy::new`] takes the
/// larger of the two formats' defaults, which is the one of the narrower format,
/// and the tiles apply to both sides as soon as one of them has a mean. The
/// fields are public for a caller that wants other numbers.
///
/// Measured on real K and V (`docs/reference/kv-real-data.md`, a context of 3.7k
/// tokens, error of attention): with 128 first and 256 newest cells exact the
/// error of the attention output at `q4` is 76% lower than with no exact cells.
/// The newest cells carry most of it (the newest 128 hold about half of the
/// attention weight) and a wider window keeps paying on that short context; of
/// the first cells, 4 give about 90% of what 128 give. What is still open is the
/// KLD of logits at a long context, where kvarn's author saw little from a window
/// beyond 128.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvPolicy {
    /// The format of the K rows of cells that are not exact.
    pub k: KvFormat,
    /// The format of the V rows of cells that are not exact.
    pub v: KvFormat,
    /// How many of the first cells of a sequence are kept in `f16`.
    pub exact_first_cells: usize,
    /// How many of the newest cells of a sequence are kept in `f16`.
    pub exact_newest_cells: usize,
}

impl KvPolicy {
    /// The policy for K stored as `k` and V as `v` with the default exact cells
    /// of the narrower of the two formats.
    pub const fn new(k: KvFormat, v: KvFormat) -> Self {
        KvPolicy {
            k,
            v,
            exact_first_cells: larger(k.default_exact_first_cells(), v.default_exact_first_cells()),
            exact_newest_cells: larger(
                k.default_exact_newest_cells(),
                v.default_exact_newest_cells(),
            ),
        }
    }

    /// Whether the cells are stored a tile at a time, which they are as soon as
    /// K or V has a format with a mean ([`KvFormat::has_mean`]).
    pub const fn has_mean(&self) -> bool {
        self.k.has_mean() || self.v.has_mean()
    }

    /// The cells of a sequence of `n_cells` cells that are stored in the formats
    /// `k` and `v`, as a range of positions; all others are exact.
    ///
    /// Without a mean these are the cells between the two windows. With one,
    /// they are the whole tiles between the two windows: the range starts and
    /// ends at a multiple of [`TILE_CELLS`]. The range is empty when the windows
    /// meet.
    pub fn quantized_cells(&self, n_cells: usize) -> Range<usize> {
        let mut start = self.exact_first_cells.min(n_cells);
        let mut end = n_cells.saturating_sub(self.exact_newest_cells);
        if self.has_mean() {
            start = start.checked_next_multiple_of(TILE_CELLS).unwrap_or(usize::MAX);
            end -= end % TILE_CELLS;
        }
        if start < end { start..end } else { 0..0 }
    }

    /// How many of the `n_cells` cells of a sequence are exact: all that are not
    /// among the [`KvPolicy::quantized_cells`].
    pub fn exact_cells(&self, n_cells: usize) -> usize {
        n_cells - self.quantized_cells(n_cells).len()
    }

    /// Whether the cell at `position` of a sequence of `n_cells` cells is exact.
    /// A position that is not in the sequence (`position >= n_cells`) is not.
    pub fn is_exact(&self, position: usize, n_cells: usize) -> bool {
        position < n_cells && !self.quantized_cells(n_cells).contains(&position)
    }

    /// Bytes that a sequence of `n_cells` cells takes in one attention layer
    /// with `kv_heads` KV heads of `head_dim` values: the exact cells at the size
    /// of an `f16` cell, the others at [`kv_bytes_per_cell`] of `k` and `v`, and
    /// for every tile of them [`kv_mean_bytes_per_tile`].
    ///
    /// `head_dim` must be a multiple of 32. A size that overflows `usize` is
    /// `usize::MAX`, as in [`kv_bytes_per_cell`].
    pub fn bytes_for_cells(
        &self,
        n_cells: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<usize, Error> {
        let exact_bytes = kv_bytes_per_cell(KvFormat::F16, KvFormat::F16, kv_heads, head_dim)?;
        let packed_bytes = kv_bytes_per_cell(self.k, self.v, kv_heads, head_dim)?;
        let mean_bytes = kv_mean_bytes_per_tile(self.k, self.v, kv_heads, head_dim)?;
        let quantized = self.quantized_cells(n_cells).len();
        // With a mean the quantized cells are whole tiles; without one there are
        // no bytes per tile.
        let tiles = quantized / TILE_CELLS;
        Ok((n_cells - quantized)
            .saturating_mul(exact_bytes)
            .saturating_add(quantized.saturating_mul(packed_bytes))
            .saturating_add(tiles.saturating_mul(mean_bytes)))
    }
}

/// Applies the 32-point Walsh-Hadamard rotation of the `qB` formats to every
/// group of `values` in place. The rotation is its own inverse, so this both
/// rotates a query and un-rotates a decoded row.
///
/// `values.len()` must be a multiple of 32; otherwise an error is returned and
/// nothing is changed. The order of operations is specified in the
/// [module documentation](self#specification), and the result is the exact
/// bits a kernel following it produces.
pub fn rotate_groups(values: &mut [f32]) -> Result<(), Error> {
    if !values.len().is_multiple_of(GROUP_VALUES) {
        return Err(Error::KvPartialGroup { values: values.len() });
    }
    let (groups, _) = values.as_chunks_mut::<GROUP_VALUES>();
    for group in groups {
        rotate_group(group);
    }
    Ok(())
}

/// The butterfly network and the final scaling, in the order the specification
/// fixes.
fn rotate_group(x: &mut [f32; GROUP_VALUES]) {
    let mut h = 1;
    while h < GROUP_VALUES {
        for i in (0..GROUP_VALUES).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    for v in x {
        *v *= INV_SQRT_32;
    }
}

/// Encodes one row of K or V values into `out`.
///
/// `values.len()` must be a multiple of 32 and `out.len()` exactly
/// `format.row_bytes(values.len())`. Every value must be finite, and for the
/// `qB` formats the rotation of no group may overflow `f32` (see the
/// [module documentation](self#what-is-defined-at-the-edges)). On any error
/// nothing is written.
pub fn encode_row(format: KvFormat, values: &[f32], out: &mut [u8]) -> Result<(), Error> {
    check_lengths(format, values.len(), out.len())?;
    check_finite(values)?;
    let Some(bits) = format.quant_bits() else {
        write_halves(values, out);
        return Ok(());
    };
    // The rotation is cheap next to an error that leaves half a row written, so
    // it runs once to find an overflow and again to encode.
    check_rotations(values)?;
    let (groups, _) = values.as_chunks::<GROUP_VALUES>();
    for (x, bytes) in groups.iter().zip(out.chunks_exact_mut(format.group_bytes())) {
        encode_group(&rotated(x), bits, bytes);
    }
    Ok(())
}

/// Encodes the rows of one tile: the rows of one head for consecutive cells,
/// `values.len() / n_values` of them (at least one), each of `n_values` values.
///
/// For `q7` to `q4` this writes the mean of the tile's rotated rows into `mean`
/// (`format.mean_bytes(n_values)` bytes: one f16 per value) and each row
/// relative to it into `out`, as "Tiles and their mean" in the
/// [module documentation](self#tiles-and-their-mean-q7-to-q4) specifies. For
/// `f16` and `q8`, which have no mean, `mean` must be empty and every row is
/// encoded as [`encode_row`] does.
///
/// `out.len()` must be the rows times `format.row_bytes(n_values)`; the rows
/// are stored one after the other. Every value must be finite and no rotation
/// may overflow, as for [`encode_row`]; the position or the group an error names
/// counts through the whole tile. On any error nothing is written.
pub fn encode_tile(
    format: KvFormat,
    n_values: usize,
    values: &[f32],
    mean: &mut [u8],
    out: &mut [u8],
) -> Result<(), Error> {
    check_tile_lengths(format, n_values, values.len(), mean.len(), out.len())?;
    check_finite(values)?;
    let Some(bits) = format.quant_bits() else {
        write_halves(values, out);
        return Ok(());
    };
    check_rotations(values)?;

    let rows = values.chunks_exact(n_values);
    let row_bytes = format.group_bytes() * (n_values / GROUP_VALUES);
    if !format.has_mean() {
        for (row, bytes) in rows.zip(out.chunks_exact_mut(row_bytes)) {
            let (groups, _) = row.as_chunks::<GROUP_VALUES>();
            for (x, bytes) in groups.iter().zip(bytes.chunks_exact_mut(format.group_bytes())) {
                encode_group(&rotated(x), bits, bytes);
            }
        }
        return Ok(());
    }

    // The mean of each group of rotated positions over the rows, then every row
    // relative to the mean as it is stored. A group is rotated once for the
    // mean and once more to encode it, so that nothing but a group is ever held.
    let n_rows = rows.len() as f32;
    let (mean_halves, _) = mean.as_chunks_mut::<2>();
    let (mean_groups, _) = mean_halves.as_chunks_mut::<GROUP_VALUES>();
    for (g, stored) in mean_groups.iter_mut().enumerate() {
        let group_of = |row: &[f32]| rotated(&row.as_chunks::<GROUP_VALUES>().0[g]);
        *stored = mean_bits(rows.clone().map(group_of), n_rows).map(u16::to_le_bytes);
    }
    for (row, bytes) in rows.zip(out.chunks_exact_mut(row_bytes)) {
        let (groups, _) = row.as_chunks::<GROUP_VALUES>();
        let stored = bytes.chunks_exact_mut(format.group_bytes());
        for ((x, means), bytes) in groups.iter().zip(&*mean_groups).zip(stored) {
            let mut centred = rotated(x);
            for (value, half) in centred.iter_mut().zip(means) {
                *value -= f16_to_f32(u16::from_le_bytes(*half));
            }
            encode_group(&centred, bits, bytes);
        }
    }
    Ok(())
}

/// The stored mean of one group of rotated positions over the rows of a tile,
/// as f16 patterns: for each position the sum of the rows' values in the order
/// of the rows, starting from `+0`, divided by the number of rows, saturated to
/// the range of f16 and rounded to the nearest one. `groups` yields the rotated
/// group of each row and `n_rows` is their number.
fn mean_bits(
    groups: impl Iterator<Item = [f32; GROUP_VALUES]>,
    n_rows: f32,
) -> [u16; GROUP_VALUES] {
    let mut sums = [0.0f32; GROUP_VALUES];
    for group in groups {
        for (sum, value) in sums.iter_mut().zip(group) {
            *sum += value;
        }
    }
    sums.map(|sum| f32_to_f16((sum / n_rows).clamp(-F16_MAX, F16_MAX)))
}

/// Decodes a stored row to the values it is stored as: for a `qB` format the
/// rotated values `code * scale`, exactly; for `f16` the values themselves.
///
/// `out.len()` is the number of values, a multiple of 32, and `bytes.len()` must
/// be exactly `format.row_bytes(out.len())`. Nothing is written on an error.
/// Attention uses this for K, with the query rotated, and for V before
/// un-rotating the accumulated result once.
pub fn decode_row_rotated(format: KvFormat, bytes: &[u8], out: &mut [f32]) -> Result<(), Error> {
    check_lengths(format, out.len(), bytes.len())?;
    let Some(bits) = format.quant_bits() else {
        let (halves, _) = bytes.as_chunks::<2>();
        for (value, half) in out.iter_mut().zip(halves) {
            *value = f16_to_f32(u16::from_le_bytes(*half));
        }
        return Ok(());
    };
    let (groups, _) = out.as_chunks_mut::<GROUP_VALUES>();
    for (bytes, values) in bytes.chunks_exact(format.group_bytes()).zip(groups) {
        decode_group(bytes, bits, values);
    }
    Ok(())
}

/// Decodes a stored row to the original domain: [`decode_row_rotated`] followed
/// by the rotation, which for `f16` does nothing.
///
/// The rotation is the only step that rounds, so this equals
/// `rotate_groups(decode_row_rotated(..))` exactly. Same lengths and same
/// guarantee of writing nothing on an error.
pub fn decode_row(format: KvFormat, bytes: &[u8], out: &mut [f32]) -> Result<(), Error> {
    decode_row_rotated(format, bytes, out)?;
    if format.is_rotated() {
        rotate_groups(out)?;
    }
    Ok(())
}

/// Decodes the rows of a tile to the values they stand for in the stored
/// domain: for `q7` to `q4` the rotated values `code * scale + mean`, for `q8`
/// `code * scale`, for `f16` the values themselves.
///
/// `out.len()` is the number of values of the tile, one or more rows of
/// `n_values`; `mean` and `bytes` must have the lengths [`encode_tile`] wrote
/// (`mean` empty for `f16` and `q8`). Nothing is written on an error.
///
/// `code * scale` is exact and the addition of the mean is one f32 addition, so
/// this has exactly one correct result. Attention does not compute it: it reads
/// the rows with [`decode_row_rotated`] and accounts for the mean once per tile
/// (see "What attention does with each width" in the
/// [module documentation](self#what-attention-does-with-each-width)). This
/// function is what that must agree with.
pub fn decode_tile_rotated(
    format: KvFormat,
    n_values: usize,
    mean: &[u8],
    bytes: &[u8],
    out: &mut [f32],
) -> Result<(), Error> {
    check_tile_lengths(format, n_values, out.len(), mean.len(), bytes.len())?;
    let row_bytes = format.group_bytes() * (n_values / GROUP_VALUES);
    let (mean_halves, _) = mean.as_chunks::<2>();
    for (bytes, row) in bytes.chunks_exact(row_bytes).zip(out.chunks_exact_mut(n_values)) {
        decode_row_rotated(format, bytes, row)?;
        // No halves for a format without a mean.
        for (value, half) in row.iter_mut().zip(mean_halves) {
            *value += f16_to_f32(u16::from_le_bytes(*half));
        }
    }
    Ok(())
}

/// Decodes the rows of a tile to the original domain: [`decode_tile_rotated`]
/// followed by the rotation of every group, which for `f16` does nothing.
/// Same lengths and same guarantee of writing nothing on an error.
pub fn decode_tile(
    format: KvFormat,
    n_values: usize,
    mean: &[u8],
    bytes: &[u8],
    out: &mut [f32],
) -> Result<(), Error> {
    decode_tile_rotated(format, n_values, mean, bytes, out)?;
    if format.is_rotated() {
        rotate_groups(out)?;
    }
    Ok(())
}

/// Checks that `n_values` is whole groups and `n_bytes` the size of their row.
fn check_lengths(format: KvFormat, n_values: usize, n_bytes: usize) -> Result<(), Error> {
    let expected = format.row_bytes(n_values)?;
    if expected != n_bytes {
        return Err(Error::KvByteLength { format, values: n_values, expected, actual: n_bytes });
    }
    Ok(())
}

/// Checks the lengths of a tile: rows of `n_values` values (whole groups, not
/// none), `n_tile_values` values in one or more whole rows, the mean as long as
/// the format has it and `n_bytes` the size of the rows.
fn check_tile_lengths(
    format: KvFormat,
    n_values: usize,
    n_tile_values: usize,
    n_mean_bytes: usize,
    n_bytes: usize,
) -> Result<(), Error> {
    let row_bytes = format.row_bytes(n_values)?;
    if n_values == 0 || n_tile_values == 0 || !n_tile_values.is_multiple_of(n_values) {
        return Err(Error::KvTileRows { values: n_tile_values, row: n_values });
    }
    let expected = format.mean_bytes(n_values)?;
    if expected != n_mean_bytes {
        return Err(Error::KvMeanLength {
            format,
            values: n_values,
            expected,
            actual: n_mean_bytes,
        });
    }
    let expected = (n_tile_values / n_values).saturating_mul(row_bytes);
    if expected != n_bytes {
        return Err(Error::KvByteLength {
            format,
            values: n_tile_values,
            expected,
            actual: n_bytes,
        });
    }
    Ok(())
}

/// Refuses a NaN or an infinity, naming the first.
fn check_finite(values: &[f32]) -> Result<(), Error> {
    let Some((position, value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) else {
        return Ok(());
    };
    let kind = if value.is_nan() {
        "NaN"
    } else if *value > 0.0 {
        "+infinity"
    } else {
        "-infinity"
    };
    Err(Error::KvNonFinite { position, kind })
}

/// Refuses a group whose rotation overflows `f32`, naming the first. `values`
/// is whole groups.
fn check_rotations(values: &[f32]) -> Result<(), Error> {
    let (groups, _) = values.as_chunks::<GROUP_VALUES>();
    match groups.iter().position(|x| rotated(x).iter().any(|v| !v.is_finite())) {
        Some(group) => Err(Error::KvRotationOverflow { group }),
        None => Ok(()),
    }
}

/// Writes `values` as f16, saturating: the `f16` format. `out` has two bytes
/// per value.
fn write_halves(values: &[f32], out: &mut [u8]) {
    let (halves, _) = out.as_chunks_mut::<2>();
    for (value, half) in values.iter().zip(halves) {
        *half = f32_to_f16(value.clamp(-F16_MAX, F16_MAX)).to_le_bytes();
    }
}

/// The rotation of one group, as a copy.
fn rotated(x: &[f32; GROUP_VALUES]) -> [f32; GROUP_VALUES] {
    let mut group = *x;
    rotate_group(&mut group);
    group
}

/// The largest code of a `bits`-bit format, `2^(bits-1) - 1`.
const fn code_max(bits: u32) -> i32 {
    (1 << (bits - 1)) - 1
}

/// The stored scale (an f16 pattern) of a group whose largest rotated magnitude
/// is `amax`: step 2 of the specification. `amax` is finite.
fn scale_bits(amax: f32, bits: u32) -> u16 {
    if amax == 0.0 {
        return 0;
    }
    let q = amax / code_max(bits) as f32;
    let nearest = f32_to_f16(q);
    // From 65520 up the nearest f16 is infinity. The scale saturates at the
    // largest finite one instead, here and again below when rounding up from
    // 65504 would reach infinity.
    if nearest >= F16_INFINITY_BITS {
        return F16_MAX_BITS;
    }
    let scale = if nearest == 0 || f16_to_f32(nearest) < q { nearest + 1 } else { nearest };
    scale.min(F16_MAX_BITS)
}

/// The reduced scales the search of `q7` to `q4` tries after the full one,
/// `2^(8-B)`: 2, 4, 8 and 16.
const fn search_steps(bits: u32) -> u32 {
    1 << (8 - bits)
}

/// The search tries the scale for `amax * (SEARCH_GRID - j) / SEARCH_GRID`.
const SEARCH_GRID: u32 = 64;

/// The sum of 32 values in the order the specification fixes: the pairs at
/// distance 16 first, then those at 8, 4, 2 and 1, which is what a warp of 32
/// lanes gets from five exchanges. Plain f32 additions, none fused with the
/// multiplication that made the values.
fn sum32(mut v: [f32; GROUP_VALUES]) -> f32 {
    let mut half = GROUP_VALUES / 2;
    while half > 0 {
        for i in 0..half {
            v[i] += v[i + half];
        }
        half /= 2;
    }
    v[0]
}

/// Encodes the rotated group `r` into `out`, `2 + 4 * bits` bytes: steps 3 and 4
/// of the specification.
fn encode_group(r: &[f32; GROUP_VALUES], bits: u32, out: &mut [u8]) {
    let amax = r.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    // A group of zeros has scale zero and codes zero.
    let (scale, codes) = if amax == 0.0 {
        (0, [0; GROUP_VALUES])
    } else if bits == 8 {
        absmax_codes(r, amax, bits)
    } else {
        searched_codes(r, amax, bits)
    };
    out[..2].copy_from_slice(&scale.to_le_bytes());
    pack_codes(&codes, bits, &mut out[2..]);
}

/// The scale and codes of `q8`: the scale from the largest magnitude and the
/// symmetric range. `amax` is the largest magnitude, above zero.
fn absmax_codes(r: &[f32; GROUP_VALUES], amax: f32, bits: u32) -> (u16, [i32; GROUP_VALUES]) {
    let scale = scale_bits(amax, bits);
    let s = f16_to_f32(scale);
    let max = code_max(bits) as f32;
    let mut codes = [0i32; GROUP_VALUES];
    for (code, v) in codes.iter_mut().zip(r) {
        *code = (v / s).round_ties_even().clamp(-max, max) as i32;
    }
    (scale, codes)
}

/// The scale and codes of `q7` to `q4`: the search of step 3 of the
/// specification. `amax` is the largest magnitude, above zero.
fn searched_codes(r: &[f32; GROUP_VALUES], amax: f32, bits: u32) -> (u16, [i32; GROUP_VALUES]) {
    let max = code_max(bits) as f32;
    // The codes run from -max - 1 to max. Their longer side is the side of the
    // larger extreme of the group: a positive scale puts it on the negative
    // values, a negative scale on the positive ones.
    let positive_max = r.iter().fold(0.0f32, |m, v| m.max(*v));
    let negative_max = r.iter().fold(0.0f32, |m, v| m.max(-*v));
    let sign = if negative_max >= positive_max { 1.0f32 } else { -1.0 };
    let sign_bit = if sign < 0.0 { 0x8000u16 } else { 0 };

    let mut best: Option<(f32, u16, [i32; GROUP_VALUES])> = None;
    for j in 0..=search_steps(bits) {
        let candidate = amax * ((SEARCH_GRID - j) as f32 / SEARCH_GRID as f32);
        let base = scale_bits(candidate, bits);
        let grid = sign * f16_to_f32(base);

        let mut codes = [0i32; GROUP_VALUES];
        let mut products = [0.0f32; GROUP_VALUES];
        let mut squares = 0i32;
        for ((code, product), v) in codes.iter_mut().zip(&mut products).zip(r) {
            let c = (v / grid).round_ties_even().clamp(-max - 1.0, max);
            *code = c as i32;
            *product = v * c;
            squares += *code * *code;
        }
        // The scale that fits these codes best in the least-squares sense,
        // rounded to the nearest f16 (and not to zero, which a code other than
        // zero rules out anyway), replaces the candidate's. With all codes zero
        // the sign means nothing and is left positive.
        let scale = if squares == 0 {
            base
        } else {
            let fitted = (sum32(products).abs() / squares as f32).min(F16_MAX);
            f32_to_f16(fitted).max(1) | sign_bit
        };

        let s = f16_to_f32(scale);
        let mut errors = [0.0f32; GROUP_VALUES];
        for ((e, v), code) in errors.iter_mut().zip(r).zip(&codes) {
            let d = v - *code as f32 * s;
            *e = d * d;
        }
        let error = sum32(errors);
        if best.is_none_or(|(least, ..)| error < least) {
            best = Some((error, scale, codes));
        }
    }
    let (_, scale, codes) = best.expect("the search has at least one candidate");
    (scale, codes)
}

/// Decodes the group in `bytes` to its rotated values.
fn decode_group(bytes: &[u8], bits: u32, out: &mut [f32; GROUP_VALUES]) {
    let s = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    let codes = unpack_codes(&bytes[2..], bits);
    for (v, code) in out.iter_mut().zip(codes) {
        *v = code as f32 * s;
    }
}

/// Writes the 32 codes as the little-endian bit stream of the layout, each as a
/// `bits`-bit two's complement number. `out` is `4 * bits` bytes.
fn pack_codes(codes: &[i32; GROUP_VALUES], bits: u32, out: &mut [u8]) {
    let mask = (1u32 << bits) - 1;
    // Bits not yet written, lowest first. At most 7 are left over when a code
    // of at most 8 bits is added, so 15 bits fit easily.
    let mut pending = 0u32;
    let mut count = 0;
    let mut next = 0;
    for &code in codes {
        pending |= (code as u32 & mask) << count;
        count += bits;
        while count >= 8 {
            out[next] = pending as u8;
            next += 1;
            pending >>= 8;
            count -= 8;
        }
    }
    debug_assert_eq!(count, 0, "32 codes of any width fill whole bytes");
}

/// Reads the 32 codes back from the bit stream, sign-extending each.
fn unpack_codes(bytes: &[u8], bits: u32) -> [i32; GROUP_VALUES] {
    let mask = (1u32 << bits) - 1;
    let mut codes = [0i32; GROUP_VALUES];
    let mut pending = 0u32;
    let mut count = 0;
    let mut next = 0;
    for code in &mut codes {
        while count < bits {
            pending |= u32::from(bytes[next]) << count;
            next += 1;
            count += 8;
        }
        let raw = pending & mask;
        pending >>= bits;
        count -= bits;
        // Move the code's sign bit to bit 31 and shift back arithmetically.
        *code = ((raw << (32 - bits)) as i32) >> (32 - bits);
    }
    codes
}

#[cfg(test)]
mod tests;
