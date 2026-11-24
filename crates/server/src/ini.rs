//! The files of options, which the configuration file of `--config` and the
//! preset file of `--models-preset` share: lines `name = value`, under
//! `[section]` headings in a preset file, and comment lines that begin with
//! `;` or `#`. A name is an option's key (its name without the dashes); what
//! a key may say is [`crate::options::of_key`]'s.

use crate::options;

/// A line `name = value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Key {
    pub name: String,
    pub value: String,
    /// The line, counted from 1.
    pub line: usize,
}

/// A line `[name]` and the keys under it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    pub line: usize,
    pub keys: Vec<Key>,
}

/// What a file holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ini {
    /// The keys before the first section: all of a configuration file's.
    pub keys: Vec<Key>,
    pub sections: Vec<Section>,
}

/// Cuts `text` into its keys and sections. A key stands once where it stands,
/// unless its option may be given several times. An error is a line and what
/// is wrong with it.
pub fn parse(text: &str) -> Result<Ini, (usize, String)> {
    let mut ini = Ini::default();
    for (index, raw) in text.lines().enumerate() {
        let (number, line) = (index + 1, raw.trim());
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            let Some(name) = rest.strip_suffix(']').map(str::trim) else {
                return Err((number, format!("'{line}' is not a section: write [name]")));
            };
            if name.is_empty() {
                return Err((number, "a section needs a name: the id of its model, or *".into()));
            }
            ini.sections.push(Section { name: name.to_owned(), line: number, keys: Vec::new() });
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            return Err((number, format!("'{line}' is not a key: write name = value")));
        };
        let (name, value) = (name.trim(), value.trim());
        let (keys, within) = match ini.sections.last_mut() {
            Some(section) => (&mut section.keys, format!(" in [{}]", section.name)),
            None => (&mut ini.keys, String::new()),
        };
        let repeats = options::find(&format!("--{name}")).is_some_and(|(opt, _)| opt.repeats);
        if let Some(first) = keys.iter().find(|key| key.name == name)
            && !repeats
        {
            return Err((
                number,
                format!("the key '{name}' stands twice{within} (first at line {})", first.line),
            ));
        }
        keys.push(Key { name: name.to_owned(), value: value.to_owned(), line: number });
    }
    Ok(ini)
}
