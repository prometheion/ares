# Reference programs

The tests of this crate compare with llama.cpp through `llama-tokenize` and through
files that llama.cpp's tools wrote. Two small programs go further: they run llama.cpp's own
code on many inputs with one start, which is what makes comparing millions of texts
possible. Both are built against a llama.cpp checkout and its build tree; the commands are
in the head of each source file. They are not built by cargo.

| Program | What it runs | Driven by |
| --- | --- | --- |
| `reftok.cpp` | `llama_tokenize` and `llama_token_to_piece` on a vocabulary loaded once (and the speed of both) | `examples/fuzz.rs` |
| `reftpl.cpp` | the Jinja interpreter of llama.cpp (`common/jinja`) on any template, without a model | `examples/probes.rs` |

The checkout the crate is verified against is the one named in `gen_unicode.py` and in
`tests/golden/generate.py`; the unicode tables and the golden files come from it. A prebuilt
`llama-tokenize` that is older than the libraries next to it crashes before it reads its
arguments, so build one from the same checkout.
