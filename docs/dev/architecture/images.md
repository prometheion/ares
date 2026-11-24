# The image tower

`ares-model::qwen3vl` holds the projector that both language models read images
through (`clip.projector_type = qwen3vl_merger`; one file per model, the same
shape with other weights and another output width): its configuration and
weight table, the pixel work in front of it, a CPU reference of the pass, and
the pass on a device (see [On the device](#on-the-device) below). The reference
is a test oracle, as the references of the language models are. Both language
models read images through it (see [Images in a prompt](#images-in-a-prompt)). It sits in `ares-model` beside `qwen35` and `qwen4exp`, with the same
parts (`config`, `weights`, `reference`, `device`).

- **Configuration and weights** are read strictly: a `clip.*` key the
  implementation does not know, another projector type, deepstack layers, an
  activation other than GELU, a merge other than 2 x 2, and a tensor that is
  missing, misshapen or not part of the projector are refused by name.
- **Pixels** (`image`): the size rule (both sides to the nearest multiple of
  32 pixels with ties to the even multiple, then scaled into the limits of 8
  and 4096 rows by default), the bicubic resize in the 22-bit fixed point of
  Pillow's resampling, `(value / 255 - mean) / std`, each operation for
  operation as llama.cpp's `mtmd` does it, and the patch rows in the merger's
  order (`patch_rows`), which the reference and the device both read. Reading
  image files belongs to the server.
- **The pass** (`reference`): patch rows of 768 values in the merger's order
  through the sum of the two patch kernels, the 48 x 48 position table resized
  bilinearly with aligned corners, 27 blocks (layer norm, one matrix for query,
  key and value, the rotation of query and key in two halves of a head by the
  patch's row and column, full softmax attention, the MLP with the tanh GELU),
  the last norm, and the merger over four patches side by side. Sums are
  accumulated in `f64` and rounded to `f32` once, in an order fixed by the
  length of the data, so the rows have the same bits on any number of threads.
  Every stage can be tapped.

**Checked against llama.cpp.** Rows of both projectors were taken from the
fork's `llama-mtmd-debug` on the CPU for eleven square patterns of 4 to 1024
rows, with the sum of every node of its graph. That program computes a noisy
version of the pass: it rounds pixels and patch kernels to f16, the input of
every BF16 matrix to bf16 and the GELU to f16 on both sides, and with flash
attention queries, keys and values to f16. The reference has a second
arithmetic with the first three of these roundings. Up to the end of the first
block its stage sums and llama.cpp's agree to 5e-6 of the summed magnitudes.
After that they drift apart, because a value that differs in its last bit can
round to the other bf16 neighbour: with the precision of the sums as the only
change (`f32` in place of `f64`), the reference with the roundings moved by
0.2 to 0.9 % on the four images tried. How much the roundings matter depends on
the image, so the test measures it per image, as the distance between the
reference with and without them (0.2 to 7 %, largest for an image of one
colour), and allows three times that plus 0.1 %. Measured on the eleven
images of both projectors: the reference with the roundings is 0.2 to 2.2 %
from llama.cpp's rows (relative root of the squared differences), the exact
one 0.2 to 6.3 %; llama.cpp with flash attention is 0.6 to 1.2 % from
llama.cpp without. The exact arithmetic is what a device is to be compared
with.

The resize could not be put through llama.cpp (its debug program reads no
image file). It equals Pillow 13 byte for byte on ten images when Pillow
filters the columns first, as llama.cpp does; one call of Pillow filters the
rows first when that leaves the smaller picture between the passes, and its
result then differs from llama.cpp's (two of the ten images, by up to 16 of
255 on a checkerboard).

The residual of the tower holds values in the thousands beside values near
one: in the 27B's projector it stays below 21 for nine blocks, is about 430
to 520 from block 9 on and about 6000 after the last block, where one MLP
writes a value that large; 8384 is the largest seen (Flash-Next's projector).
Sums of such values need `f32`.

On the 16 threads of the development machine, under other load, the pass takes
32 to 37 s for 256 rows and 115 to 127 s for 1024.

### On the device

`qwen3vl::DeviceTower` runs the pass on one device: an image's patch rows in
(`image::patch_rows`, in the merger's order), the merged rows of the language
model's width out, in a buffer on the device that the language model can read
(see [Images in a prompt](#images-in-a-prompt)).

- **Products** are the tile kernels of the language models (`Kernels::matmul`)
  on their f16 path: f16 operands, f32 sums. The activations are prepared in
  that path's form (each row divided by the power of two above its largest
  value, so a row of thousands keeps its precision). `ffn_down` has 4304
  inputs, which is not a multiple of 32: its rows get 16 zero columns at load
  and the GELU in front of it writes 16 zeros. The patch matrix is the sum of
  the file's two kernels, formed once in `f32` (a matrix of 768 inputs; no
  convolution).
- **Everything else** is a module of its own (`kernels/tower.cu`,
  `ares_cuda::TowerKernels`): the layer norm with its bias and the bias with
  the tanh GELU, each writing the next product's f16 input in the launch that
  computes it; the residual add in `f32` with the product's bias (the residual
  reaches 5400 to 8400, see above); the patches' bias and the position table
  resized to the grid, every operation the reference's in its order without
  contraction, so these are its bits; query, key and value with their bias,
  query and key rotated by the patch's row and column (the angles built by
  multiplying on in `f32` as the reference does), as f16 rows per head; and the
  attention.
- **Attention** of every patch over every patch (up to 16384 patches for 4096
  rows) has no mask and no pages, so it is a kernel of its own in the shape of
  FlashAttention 2: a block holds 128 queries of one head (8 warps of 16) in
  registers and walks the keys in chunks of 64, the next chunk copied into
  shared memory while this one is computed; the head of 72 values is padded to
  80 for the products, sums are f32, the softmax is online in f32, the
  probabilities are rounded to f16 for the second product and summed as
  rounded.
- **Groups.** A block's work before and after its attention is row by row, so
  it runs for 2048 patches at a time; what grows with the image is the
  residual, query, key and value and the attention's output (16.5 KiB a
  patch). The scratch of an image of 4096 rows is 336.6 MiB, of 1024 rows
  133 MiB (its region of the attention's output also takes the merged rows at
  the end of a pass, which are wider: 8 MiB more at 4096 rows; measured before
  that, 328.6 and 131 MiB). Groups of 4096 patches were 1.6 to 3.8 % faster on both cards (68
  MiB more scratch from 1024 rows on); the VRAM is worth more to the language
  model.

**Two homes for the weights** (`qwen3vl::Home`, a load option; llama.cpp's
`mmproj` with and without `no-mmproj-offload`). The weights are one image of
bytes in the device's layout, cut into units: the stem (patch matrix, bias,
position table), each of the 27 blocks (29.1 MiB), and the merger's two
matrices in parts of whole tiles of output rows no larger than a block (2 and
2 parts for the 27B's projector, 2 and 1 for Flash-Next's).

- *VRAM*: the image is uploaded at load and every unit is read where it lies.
- *RAM* (the default): the image lies in page-locked RAM, which the model that
  carries the tower books as required memory inside `--ram`
  (`DeviceTower::pinned_bytes`, added to `Model::pinned_ram`). Every image
  uploads it unit by unit into two slots of VRAM on a copy stream of the
  tower's own: unit `u + 2` goes into the slot of unit `u` once the pass has
  recorded that it is done with it, and the pass waits for a unit's upload
  before its first launch, so the upload of the next unit runs while the
  current one computes. Between images it holds the two slots and nothing
  else.

Both homes launch the same kernels on the same units, so their rows are the
same bit for bit (the tests compare them).

**Checked** against the exact reference (`tests/qwen3vl_device.rs`), with a
tolerance measured per image as the reference's own tests measure theirs: the
reference has a third arithmetic, `HalfOperands`, with the device's roundings
(the inputs of every product in the f16 form above, the weights to f16,
queries, keys, values and probabilities to f16); its distance from the exact
rows says how far such roundings carry the image, and the device may be three
times that far plus 0.1 %. The cases are the eleven square patterns of the
reference dumps and the ten own images of the Pillow check (wide, tall, below
the minimum, and one shrunk to 4060 rows), resized here; the reference's rows
are kept in a directory, since an image of 4060 rows takes it half an hour on
12 threads of the CPU. Measured on both cards (2026-11-16, both projectors, all 21 cases up
to 4060 rows): the device is 3.0e-4 to 6.7e-3 from the exact rows, 0.62 to 1.8
times as far as the reference with the device's roundings is, so no case used
more than 37 % of its tolerance (the image of 4060 rows: 1.05e-3 for the 27B's
projector and 1.39e-3 for Flash-Next's, against 1.08e-3 and 1.33e-3); the two
homes gave the same rows bit for bit, and so did the two cards.

The residual after the stem and after the first two blocks is checked value by
value as well: 0.9e-4 to 1.7e-4 of the exact reference after the stem (the f16
operands of the patch product), 2.1e-4 to 3.2e-4 after block 0 and 3.6e-4 to
7.5e-4 after block 1, the same on both cards and in both homes. The kernels
have tests of their own (`crates/cuda/tests/tower.rs`): the f16 rows the norm
and the GELU write are the tile kernels' form byte for byte, the residual, the
placing of parts and the positions are the host's `f32` bit for bit, the
rotation is within an f16 step and the attention within 2e-3 of the host's in
`f64`.

### Images in a prompt

An image goes from a request to the rows of a step in five places, each with
one job.

- **The routes** (`ares-server`, `images.rs`) decode the image file: an
  `image_url` part of a chat message (a `data:image/...;base64,` URL, or base64
  alone, as llama-server takes it), an Anthropic `image` block (base64) in a
  turn of the user or in the result of a tool, a Responses `input_image` (a data
  URL). JPEG, PNG, GIF (the first frame), WebP and BMP become 8-bit RGB, alpha
  dropped and no EXIF orientation applied, as llama.cpp's `stb_image` does; at
  most 16384 pixels a side and 64 megapixels, read from the header before
  anything is decoded. A remote URL is refused with a 400 that says to send
  the image itself (fetching is backlog). The Anthropic and Responses routes turn
  their images into chat parts, so one parser reads them all. A message with
  images is joined as llama-server joins it (a line break between two text
  parts, none next to an image), and each image is kept at its byte in the
  text (`ares_api::Image`).
- **The model** (`ares-model::chat`, `qwen3vl::Projector`) writes the prompt as
  llama.cpp's `mtmd` does: a character of the private use area stands for each
  image while the chat template renders the conversation, the text is cut
  there, each piece is tokenized with its special tokens, and an image becomes
  `<|vision_start|>`, one `<|image_pad|>` per row, `<|vision_end|>`. The
  projector resizes it by the size rule (`--image-max-tokens`, 8 to 4096 rows,
  4096 by default) and names it: the SHA-256 of the projector file's identity,
  the resized size and the pixels. Counting a prompt (`count_tokens`,
  `/tokenize`) takes the rows from the size rule without resizing. The places
  where the parts of a conversation end are found on the rendering with one pad
  per image and moved behind the rows.
- **The conductor** (`ares-runtime`, see "Images" in its module) keeps each
  request's images (`ImageSpan`, `Spans`) and the names of its cells. The image
  tower is a job of the device that holds it (`Executor::tower`,
  `DeviceModel::tower`), run between steps as prompt work: a piece of prompt
  starts with as many units of the pass (the stem, the 27 blocks, the merger's
  parts) as fit the time of a piece while sequences generate, by what the
  units took before, and with the rest of the pass when none does; its time is
  charged to the share as a piece's. A prompt is read up to the first image
  whose rows are not in the store (`ImageStore`, by name, least recently used
  first, `RuntimeConfig::image_rows`, 256 MiB of ordinary memory), and a step
  takes at most as many image rows as the image stage holds (page-locked,
  `min(max_rows, 1024)` rows of the model's width, inside `--ram`). The record
  says which rows take staged rows (its image runs), and every row carries its
  rotary position: `(p, p, p)` for text, `(p, p + j / nx, p + j % nx)` for
  row `j` of an image at `p`, the text after an image `max(nx, ny)` on (the
  row's cell stays in the record's row extras).
- **The part that embeds** (`blocks::rows::splice`) copies the staged rows from
  the page-locked stage over the rows it gathered, before the first layer reads
  them: on one device, the first device of a layer split, the device that
  embeds in a tensor split (`qwen35`), and the part that embeds of Flash-Next,
  also for each micro-batch of a prompt read layer by layer (see "Long prompts
  layer by layer"). The records a part builds itself (the groups of a split,
  Flash-Next's step record and a pass's micro-batches) carry the rows' rotary
  positions. The n-gram embedding of Flash-Next
  hashes the pad token of an image cell, which is `ple.image_token_id` (the
  model refuses a projector when the two differ).
- **The prefix cache** names an image's cells by the image and the row
  (`cell_name`: the top bit set, 63 bits mixed from the image's name and the
  row), in the 64-bit names the chain hashes, the tails, the RAM level and
  the store on SSD take; a text cell's name is its token, so the pages of text
  are named as before. A conversation that sends its image again with every
  turn resumes from the cache, and a replay of cached image cells takes their
  rows from the store, or runs the tower again when the store let them go.

**Where the tower runs.** On one device, chosen by the server from what the
devices report: with its weights in RAM (`--mmproj-weights ram`, the default)
the one with the fastest link to the host (`current_link_speed` times
`current_link_width` from sysfs), then the most multiprocessors; with its
weights in VRAM the most multiprocessors first. Which device computes a block
faster is not measured at load. Its VRAM (`Projector::vram_bytes`: the two
slots or all the weights, and the scratch of the largest image, 337 MiB at
4096 rows) is taken off that device before anything is planned there, so the
context or the expert slots give it up, and the part reserves it in its ledger
at load. The scratch's last region also takes the image's rows at the end of
the pass, which are read back to the host from there.

**The drafters.** The MTP block writes its keys and runs the links of its
chains at the rotary positions of the step's rows, as the model's attention
does (`blocks::mtp::Driver::set_rotary`), but it reads an image cell as the pad
token's embedding, where llama.cpp hands the draft context the image's rows;
the DFlash drafter takes the cells as positions. Only the acceptance of drafts
after an image can suffer: what is accepted is the main model's pick.

**Not built.** Images are not fetched from URLs, and the rows of the store are
not kept on SSD (a conversation resumed after a restart runs the tower once
for each image whose cells are replayed).
