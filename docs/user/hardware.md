# Hardware

Ares runs on Linux with NVIDIA cards through CUDA. It loads the driver's
`libcuda.so.1` at run time; the CUDA toolkit is needed to build, not to run.
[models.md](models.md#what-was-run) lists the setups that were run.

## GPUs

CUDA runs only on the GPU classes Ares was tested on.

| Class | Compute capability | Cards | CUDA |
| --- | --- | --- | --- |
| Ampere | 8.6 | RTX 30 | runs |
| Ada | 8.9 | RTX 40 | runs |
| Blackwell | 12.0 | RTX 50 | refused until it has been tested |
| any other (8.0, 8.7, 9.0, below 8.0) | | | refused |

A device of another class is refused when it is opened: the server skips it and
logs why, and the load fails only when no device is left. Code for 8.6 does not
run an 8.7 device. Every other GPU is meant to run through the Vulkan backend
later: its kernels exist, but no model runs on Vulkan yet.

`ARES_CUDA_ARCHS` sets the classes a build makes machine code for, such as `89`
for one RTX 40 card. The default is the tested classes, `86,89`. A class that is
not tested can be named on purpose and the build warns; a class below 8.0 is
an error, and so is a class the installed `nvcc` does not list.

## One card or two

Ares was tested with one card and with two.

| Option | What it does |
| --- | --- |
| `--devices 0,1` | the CUDA devices to use, by the driver's index; default all |
| `--split single\|tensor\|layer` | one device, a slice of every block on each device, or a range of blocks on each device |
| `--shares 0.625,0.375` | each device's share of the weights under a split (27B only); default by the memory bandwidth each device reports |

The defaults are in [options.md](options.md#where-the-model-goes). On
Flash-Next, `--split tensor` takes exactly two devices, and how the two share a
layer is measured at load. The start log says what each device holds.

## VRAM

A load plans what it takes of each device against what the device has free
then, less what the engine leaves there, and the start log says for every
device what a plan may take and why:

| What is left | When | How much |
| --- | --- | --- |
| A cap | `--vram 10G`, or `--vram 0:10G,1:11G` for each device | The engine holds no more than this on the device, everything the driver counts for it there (what `nvidia-smi` shows for the process). |
| A reserve for a display | The device drives a display: NVML says a display is active on it, or Linux's DRM lists a connected display of the card | A sixteenth of the device's memory, or 32 frames of its displays (4 bytes a pixel of their preferred modes), whichever is more: about 750 MiB on a 12 GiB card with a 1080p display, about 1 GiB on an 8 GiB card with a 4K display. |
| A reserve for other programs | Other processes hold memory on the device when a model loads | As much as they hold, at most a sixteenth of the device. |

Without a cap and on a device that drives no display and that nothing else
uses, nothing is held back but 256 MiB for the driver beside the engine's own
buffers. A cap replaces the reserve on its device: what is left is the
operator's choice. Without NVML (`libnvidia-ml.so.1`, which comes with the
driver) a cap is refused and only DRM tells of a display.

## Peer access between cards

At load the log names every link between two devices: `peer access` or `no
peer access`. With peer access on every pair, data goes directly from device to
device. Without it, the devices exchange through page-locked host memory, which
is slower for a tensor split, and Flash-Next takes a range of layers on each
device by default instead of both devices on every layer.

Peer access between consumer cards needs a driver build that allows it: its
library and its kernel modules. `ARES_LIBCUDA=<path>` loads such a library in
place of the system's `libcuda.so.1`. The log names the library it loaded.

## RAM

`--ram` is one upper limit for all page-locked host memory of the engine, for
all loaded models together. The default is all the machine has.

| Part | What it holds |
| --- | --- |
| What the engine needs to run | step buffers, the token embedding table (27B), the image tower's weights in RAM, one checkpoint place, and with `--cache` the pages on their way to the SSD. It is taken first; a limit below it is refused with every share named. |
| The tier of experts | Flash-Next: the experts that have no slot in VRAM. It takes the rest, and gives memory up for the K and V of a long context beyond VRAM. |
| The prompt cache in RAM | K and V that left the devices, and checkpoints of recurrent state; it takes what the tier leaves. |

Below the limit the engine holds what the machine has available less a small
headroom. When other programs need memory, it gives memory back between steps,
the prompt cache first, and takes it again when it is free.

## SSD

| Use | How |
| --- | --- |
| Model weights | read with io_uring, with direct I/O where the file system allows it (else through the page cache) |
| Flash-Next experts | the experts that neither VRAM nor RAM holds are read from the model file while the model runs; a model larger than RAM runs this way |
| Prompt cache | `--cache <DIR>` keeps computed state on the SSD with direct I/O, at most `--cache-size` (default `32G`), so a conversation resumes after an unload and a restart |

Flash-Next computes experts that are missing on the card on the CPU, from their
bytes in RAM, with AVX2 and AVX-VNNI kernels picked at run time and portable
code on older CPUs.
