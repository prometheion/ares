//! What a load reads besides the model file, before a device is touched: the
//! drafters' files, the draft vocabulary, the image projector and the chat
//! template. A file that is not what its option says is refused by what it is.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ares_core::units::Bytes;
use ares_gguf::Gguf;
use ares_model::qwen3vl::{self, Projector};
use ares_tokenizer::ChatOptions;

use super::{LoadError, refuse};
use crate::plan::{self, Drafter, LoadOptions, ProjectorWeights};

/// Whether the model's MTP block takes part in speculation: when
/// `--drafters` names `mtp`, or names nothing and the model has one
/// (`has_block`). `--drafters mtp` for a model with none is refused, saying
/// `no_block`.
pub(crate) fn mtp_takes_part(
    options: &LoadOptions,
    has_block: bool,
    no_block: &str,
) -> Result<bool, LoadError> {
    let Some(list) = &options.drafters else {
        return Ok(has_block);
    };
    let asked = list.contains(&Drafter::Mtp);
    if asked && !has_block {
        return Err(refuse(format!("--drafters mtp: {no_block}")));
    }
    Ok(asked)
}

/// The drafters from files of their own that `--drafters` names, each
/// recognised by the architecture its file names: an MTP block of the model
/// (`mtp_architecture`, the block of a model whose file carries none or
/// another one) and a DFlash drafter (architecture `dflash`), one of each at
/// most. A file that is no drafter of this model is refused by what it is.
pub(crate) struct DrafterFiles {
    pub mtp: Option<(PathBuf, Gguf)>,
    pub dflash: Option<Arc<ares_model::dflash::File>>,
}

pub(crate) fn drafter_files(
    options: &LoadOptions,
    mtp_architecture: &str,
) -> Result<DrafterFiles, LoadError> {
    let mut files = DrafterFiles { mtp: None, dflash: None };
    for drafter in options.drafters.iter().flatten() {
        let Drafter::File(path) = drafter else { continue };
        let named = |e: String| refuse(format!("--drafters {}: {e}", path.display()));
        let gguf = Gguf::open(path).map_err(|e| named(e.to_string()))?;
        match gguf.architecture() {
            Some(ares_model::dflash::ARCHITECTURE) if files.dflash.is_none() => {
                let file =
                    ares_model::dflash::File::open(path).map_err(|e| named(e.to_string()))?;
                files.dflash = Some(Arc::new(file));
            }
            Some(arch) if arch == mtp_architecture && files.mtp.is_none() => {
                files.mtp = Some((path.clone(), gguf));
            }
            Some(arch) if arch == ares_model::dflash::ARCHITECTURE || arch == mtp_architecture => {
                return Err(refuse("--drafters: one MTP block and one DFlash drafter at most"));
            }
            other => {
                return Err(named(format!(
                    "a file of architecture '{}' is no drafter of this model (an MTP block is \
                     '{mtp_architecture}', a DFlash drafter 'dflash')",
                    other.unwrap_or("none")
                )));
            }
        }
    }
    if files.mtp.is_some() && options.drafters.iter().flatten().any(|d| *d == Drafter::Mtp) {
        return Err(refuse(
            "--drafters: mtp names the MTP block of the model file and a file names another; \
             the model drafts with one",
        ));
    }
    Ok(files)
}

/// The token ids of the draft vocabulary file `--draft-vocab` names
/// ([`ares_model::files::read_draft_vocabulary`]); an empty list is refused.
pub(crate) fn read_vocabulary(path: &Path) -> Result<Arc<[u32]>, LoadError> {
    let tokens = ares_model::files::read_draft_vocabulary(path)
        .map_err(|e| refuse(format!("--draft-vocab {}: {e}", path.display())))?;
    if tokens.is_empty() {
        return Err(refuse(format!("--draft-vocab {}: the list is empty", path.display())));
    }
    Ok(tokens.into())
}

/// Opens the projector of `--mmproj` for a model of width `n_embd`, with the
/// limits and the home of the options; read and checked before a device is
/// touched.
pub(crate) fn open_projector(
    options: &LoadOptions,
    n_embd: usize,
) -> Result<Option<Projector>, LoadError> {
    let Some(path) = &options.mmproj else {
        if options.mmproj_weights != ProjectorWeights::Ram
            || options.image_max_tokens != plan::IMAGE_MAX_TOKENS
        {
            return Err(refuse(
                "--mmproj-weights and --image-max-tokens are for a projector, and --mmproj names \
                 none",
            ));
        }
        return Ok(None);
    };
    let failed = |what: String| refuse(format!("--mmproj {}: {what}", path.display()));
    let gguf = Gguf::open(path).map_err(|e| failed(e.to_string()))?;
    if gguf.architecture() != Some("clip") {
        return Err(failed(format!(
            "the file is not a projector (its architecture is {})",
            gguf.architecture().unwrap_or("not declared")
        )));
    }
    let limits = qwen3vl::Limits {
        min_rows: plan::IMAGE_MIN_TOKENS as usize,
        max_rows: options.image_max_tokens as usize,
    };
    let home = match options.mmproj_weights {
        ProjectorWeights::Ram => qwen3vl::Home::Ram,
        ProjectorWeights::Vram => qwen3vl::Home::Vram,
    };
    let projector = Projector::open(gguf, limits, home).map_err(|e| failed(e.to_string()))?;
    if projector.n_out() as usize != n_embd {
        return Err(failed(format!(
            "its rows have {} values and the model's {n_embd}: it is another model's projector",
            projector.n_out()
        )));
    }
    tracing::info!(
        path = %path.display(),
        "images: the projector's tower ({} blocks), images of {} to {} rows, its weights in {} \
         ({} of VRAM and {} of page-locked RAM)",
        projector.config().n_layer,
        limits.min_rows,
        limits.max_rows,
        match home {
            qwen3vl::Home::Ram => "RAM, uploaded block by block for every new image",
            qwen3vl::Home::Vram => "VRAM",
        },
        Bytes(projector.vram_bytes()),
        Bytes(projector.pinned_bytes()),
    );
    Ok(Some(projector))
}

/// The settings a model renders conversations with: the variables of the
/// chat template, and the template of `--chat-template-file` in place of the
/// file's, compiled with the model's tokenizer. Also the text of the template
/// that is in effect, if there is one.
pub(crate) fn chat_options(
    options: &LoadOptions,
    tokenizer: &ares_tokenizer::Tokenizer,
) -> Result<(ChatOptions, Option<String>), LoadError> {
    let mut chat =
        ChatOptions { variables: options.chat_variables.clone(), ..ChatOptions::default() };
    let Some(path) = &options.chat_template else {
        return Ok((chat, tokenizer.chat_template_source().map(str::to_owned)));
    };
    let failed = |what: String| {
        LoadError::Options(format!("--chat-template-file '{}': {what}", path.display()))
    };
    let source = std::fs::read_to_string(path).map_err(|e| failed(e.to_string()))?;
    let template = tokenizer.compile_template(&source).map_err(|e| failed(e.to_string()))?;
    tracing::info!(file = %path.display(), "the chat template of the file takes the place of the model's");
    chat.template = Some(Arc::new(template));
    Ok((chat, Some(source)))
}
