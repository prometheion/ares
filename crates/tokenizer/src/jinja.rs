//! The Jinja dialect of chat templates: what is supported, and how.
//!
//! Chat templates are written for Python's Jinja and run by llama.cpp's own
//! interpreter, which differs from both Python and MiniJinja in many small
//! ways: `{{ true }}` prints `True`, `{{ none }}` prints nothing, `|trim` strips
//! the ASCII white space only, `|length` of a string counts bytes. A prompt
//! that differs in one character is a different model input, so this module
//! does not run templates with the defaults of MiniJinja. It runs them in an
//! environment that has only the filters, tests, functions and methods defined
//! below, each written after llama.cpp's, and it refuses a template at load
//! time that uses anything else in a way that would run.
//!
//! What the supported dialect covers is what the chat templates of the Qwen3.5
//! family use: `if`, `for`, `set` (also on a `namespace`), `macro`, string
//! concatenation, comparisons, `and` `or` `not`, `in`, slices, conditional
//! expressions, and the names in [`FILTERS`], [`TESTS`] and [`METHODS`].

use std::collections::HashSet;
use std::fmt::Write as _;

use minijinja::machinery::ast::{BinOpKind, CallArg, Expr, Stmt};
use minijinja::value::{Kwargs, Rest, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State, UndefinedBehavior, Value};

use crate::json::{float_json, float_repr, write_string};

/// The filters a template may use.
const FILTERS: &[&str] = &["default", "items", "length", "safe", "string", "tojson", "trim"];
/// The tests a template may use.
const TESTS: &[&str] =
    &["defined", "false", "iterable", "mapping", "none", "sequence", "string", "true", "undefined"];
/// The methods a template may call on a string or a mapping.
const METHODS: &[&str] = &[
    "endswith",
    "get",
    "items",
    "keys",
    "lstrip",
    "rsplit",
    "rstrip",
    "split",
    "startswith",
    "strip",
    "values",
];
/// The functions a template may call besides its own macros.
const FUNCTIONS: &[&str] = &["namespace", "raise_exception"];
/// Names that llama.cpp resolves to a method of the value, so that
/// `schema.items` is the method and not the key `items`. A template that reads
/// one of them as a plain attribute would mean something else here.
const METHOD_NAMES_AS_ATTRIBUTES: &[&str] = &[
    "append",
    "capitalize",
    "default",
    "dictsort",
    "endswith",
    "first",
    "float",
    "format",
    "get",
    "indent",
    "int",
    "items",
    "join",
    "keys",
    "last",
    "length",
    "list",
    "lower",
    "lstrip",
    "map",
    "max",
    "min",
    "pop",
    "reject",
    "rejectattr",
    "replace",
    "reverse",
    "rsplit",
    "rstrip",
    "safe",
    "select",
    "selectattr",
    "slice",
    "sort",
    "split",
    "startswith",
    "string",
    "strip",
    "title",
    "tojson",
    "unique",
    "upper",
    "values",
    "abs",
];
/// Variables that llama.cpp sets from the clock, which a template of ours does not get.
const CLOCK_VARIABLES: &[&str] = &["datetime", "date_string", "strftime_now"];

/// Something in a template that this dialect does not support.
#[derive(Debug)]
pub(crate) struct Unsupported {
    pub(crate) what: String,
    pub(crate) line: u32,
}

/// Looks through the template for constructs outside the dialect and returns
/// the text to compile: the template with its `list[::-1]` rewritten to
/// `(list|reverse)`.
///
/// MiniJinja panics on a backwards slice of an empty list or string, and a panic
/// in the server's step loop is not an answer to a request. The `reverse` filter
/// below has the same result and no such case.
pub(crate) fn prepare(source: &str) -> Result<String, Vec<Unsupported>> {
    let syntax = Default::default();
    let whitespace = minijinja::machinery::WhitespaceConfig {
        keep_trailing_newline: false,
        lstrip_blocks: true,
        trim_blocks: true,
    };
    let ast = match minijinja::machinery::parse(source, "template", syntax, whitespace) {
        Ok(ast) => ast,
        // A syntax error is found again, with a better message, when the
        // environment compiles the template.
        Err(_) => return Ok(source.to_owned()),
    };
    let mut check = Check {
        source,
        problems: Vec::new(),
        macros: HashSet::new(),
        namespaces: HashSet::new(),
        edits: Vec::new(),
    };
    check.collect_names(&ast);
    check.stmt(&ast);
    if !check.problems.is_empty() {
        return Err(check.problems);
    }
    let mut rewritten = source.to_owned();
    let mut edits = check.edits;
    edits.sort_by_key(|edit| std::cmp::Reverse((edit.start, edit.end)));
    for edit in edits {
        rewritten.replace_range(edit.start..edit.end, edit.text);
    }
    Ok(rewritten)
}

/// A change to the text of a template.
struct Edit {
    start: usize,
    end: usize,
    text: &'static str,
}

struct Check<'a> {
    source: &'a str,
    problems: Vec<Unsupported>,
    macros: HashSet<String>,
    /// Variables that hold a `namespace`, whose attributes are the ones assigned.
    namespaces: HashSet<String>,
    edits: Vec<Edit>,
}

impl Check<'_> {
    fn flag(&mut self, what: impl Into<String>, span: minijinja::machinery::Span) {
        self.problems.push(Unsupported { what: what.into(), line: u32::from(span.start_line) });
    }

    /// The names the template defines as macros and as namespaces, wherever they are.
    fn collect_names(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Template(t) => t.children.iter().for_each(|s| self.collect_names(s)),
            Stmt::Macro(m) => {
                self.macros.insert(m.name.to_string());
                m.body.iter().for_each(|s| self.collect_names(s));
            }
            Stmt::ForLoop(f) => {
                f.body.iter().chain(&f.else_body).for_each(|s| self.collect_names(s))
            }
            Stmt::IfCond(i) => {
                i.true_body.iter().chain(&i.false_body).for_each(|s| self.collect_names(s))
            }
            Stmt::Set(set) => {
                if let (Expr::Var(var), Expr::Call(call)) = (&set.target, &set.expr)
                    && matches!(&call.expr, Expr::Var(f) if f.id == "namespace")
                {
                    self.namespaces.insert(var.id.to_string());
                }
            }
            _ => {}
        }
    }

    fn body(&mut self, body: &[Stmt]) {
        for stmt in body {
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Template(t) => self.body(&t.children),
            Stmt::EmitRaw(_) => {}
            Stmt::EmitExpr(e) => self.expr(&e.expr),
            Stmt::ForLoop(f) => {
                if f.recursive {
                    self.flag("a recursive for loop", f.iter.span());
                }
                self.target(&f.target);
                self.expr(&f.iter);
                if let Some(filter) = &f.filter_expr {
                    self.expr(filter);
                }
                self.body(&f.body);
                self.body(&f.else_body);
            }
            Stmt::IfCond(i) => {
                self.expr(&i.expr);
                self.body(&i.true_body);
                self.body(&i.false_body);
            }
            Stmt::Set(s) => {
                match &s.target {
                    Expr::GetAttr(attr) if matches!(attr.expr, Expr::Var(_)) => {}
                    target => self.target(target),
                }
                self.expr(&s.expr);
            }
            Stmt::Macro(m) => {
                for default in &m.defaults {
                    self.expr(default);
                }
                self.body(&m.body);
            }
            Stmt::SetBlock(s) => self.flag("a set block ({% set x %}...{% endset %})", s.span()),
            Stmt::WithBlock(w) => self.flag("a with block", w.span()),
            Stmt::AutoEscape(a) => self.flag("an autoescape block", a.span()),
            Stmt::FilterBlock(f) => self.flag("a filter block", f.span()),
            Stmt::CallBlock(c) => self.flag("a call block", c.span()),
            Stmt::Do(d) => self.flag("a do statement", d.span()),
            // Imports, includes, inheritance and loop controls exist only when
            // MiniJinja is built with the features for them; this crate does not.
            #[allow(unreachable_patterns)]
            _ => self.flag(
                "a statement of a kind that is not supported",
                minijinja::machinery::Span::default(),
            ),
        }
    }

    /// The target of a `for` or a `set`: a name, or names to unpack into.
    fn target(&mut self, target: &Expr) {
        match target {
            Expr::Var(_) => {}
            Expr::List(list) if list.items.iter().all(|i| matches!(i, Expr::Var(_))) => {}
            other => self.flag("an assignment to something other than names", other.span()),
        }
    }

    fn args(&mut self, args: &[CallArg]) {
        for arg in args {
            match arg {
                CallArg::Pos(e) | CallArg::Kwarg(_, e) => self.expr(e),
                CallArg::PosSplat(e) | CallArg::KwargSplat(e) => {
                    self.flag("an argument list with * or **", e.span())
                }
            }
        }
    }

    fn expr(&mut self, expr: &Expr) {
        let span = expr.span();
        match expr {
            Expr::Var(var) => {
                if CLOCK_VARIABLES.contains(&var.id) {
                    self.flag(format!("'{}', which depends on the clock", var.id), span);
                }
            }
            Expr::Const(_) => {}
            Expr::Slice(s) => {
                self.expr(&s.expr);
                let reversed = s.start.is_none()
                    && s.stop.is_none()
                    && s.step.as_ref().and_then(Expr::as_const) == Some(Value::from(-1));
                let (outer, inner) = (span, s.expr.span());
                let (inner_end, outer_end) = (inner.end_offset as usize, outer.end_offset as usize);
                let tail = self
                    .source
                    .get(inner_end..outer_end)
                    .map(|t| t.split_whitespace().collect::<String>());
                if reversed
                    && inner.start_offset == outer.start_offset
                    && tail.as_deref() == Some("[::-1]")
                {
                    self.edits.push(Edit {
                        start: outer.start_offset as usize,
                        end: outer.start_offset as usize,
                        text: "(",
                    });
                    self.edits.push(Edit { start: inner_end, end: outer_end, text: "|reverse)" });
                } else {
                    self.flag("a slice other than list[::-1]", span);
                }
            }
            Expr::UnaryOp(u) => self.expr(&u.expr),
            Expr::BinOp(b) => {
                if matches!(
                    b.op,
                    BinOpKind::Div | BinOpKind::FloorDiv | BinOpKind::Rem | BinOpKind::Pow
                ) {
                    self.flag("an arithmetic operator other than + - *", span);
                }
                self.expr(&b.left);
                self.expr(&b.right);
            }
            Expr::Compare(c) => {
                if c.ops.len() > 1 {
                    self.flag("a chained comparison (a < b < c)", span);
                }
                self.expr(&c.expr);
                for op in &c.ops {
                    self.expr(&op.expr);
                }
            }
            Expr::IfExpr(i) => {
                self.expr(&i.test_expr);
                self.expr(&i.true_expr);
                match &i.false_expr {
                    Some(e) => self.expr(e),
                    None => self.flag("a conditional expression without else", span),
                }
            }
            Expr::Filter(f) => {
                if !FILTERS.contains(&f.name) {
                    self.flag(format!("the filter '{}'", f.name), span);
                }
                if let Some(e) = &f.expr {
                    self.expr(e);
                }
                self.args(&f.args);
            }
            Expr::Test(t) => {
                if !TESTS.contains(&t.name) {
                    self.flag(format!("the test '{}'", t.name), span);
                }
                self.expr(&t.expr);
                self.args(&t.args);
            }
            Expr::GetAttr(a) => {
                // The `loop` object and a namespace have no methods in llama.cpp, so
                // `loop.first` and `ns.last` are the attributes.
                let plain = matches!(&a.expr, Expr::Var(v) if v.id == "loop" || self.namespaces.contains(v.id));
                if !plain && METHOD_NAMES_AS_ATTRIBUTES.contains(&a.name) {
                    self.flag(
                        format!(
                            "reading '.{}' without calling it, which llama.cpp reads as a method",
                            a.name
                        ),
                        span,
                    );
                }
                self.expr(&a.expr);
            }
            Expr::GetItem(g) => {
                self.expr(&g.expr);
                self.expr(&g.subscript_expr);
            }
            Expr::Call(c) => {
                match &c.expr {
                    Expr::Var(f) => {
                        if !FUNCTIONS.contains(&f.id) && !self.macros.contains(f.id) {
                            self.flag(format!("the function '{}'", f.id), span);
                        }
                        // llama.cpp binds the arguments of a call by name and by
                        // place in a way of its own when both are mixed.
                        if self.macros.contains(f.id)
                            && c.args.iter().any(|a| matches!(a, CallArg::Kwarg(..)))
                        {
                            self.flag(
                                format!("keyword arguments in a call of the macro '{}'", f.id),
                                span,
                            );
                        }
                        if CLOCK_VARIABLES.contains(&f.id) {
                            self.flag(format!("'{}', which depends on the clock", f.id), span);
                        }
                    }
                    Expr::GetAttr(method) => {
                        if !METHODS.contains(&method.name) {
                            self.flag(format!("the method '.{}()'", method.name), span);
                        }
                        self.expr(&method.expr);
                    }
                    other => self
                        .flag("a call of something other than a function or method", other.span()),
                }
                self.args(&c.args);
            }
            Expr::List(l) => l.items.iter().for_each(|i| self.expr(i)),
            Expr::Map(_) => self.flag("a dictionary literal", span),
        }
    }
}

/// The environment templates run in.
pub(crate) fn environment() -> Environment<'static> {
    let mut env = Environment::empty();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_keep_trailing_newline(false);
    // llama.cpp lets an attribute of an undefined value be undefined again.
    env.set_undefined_behavior(UndefinedBehavior::Chainable);
    env.set_formatter(format_value);
    env.set_unknown_method_callback(call_method);

    env.add_function("namespace", minijinja::functions::namespace);
    env.add_function("raise_exception", raise_exception);

    env.add_filter("default", default);
    env.add_filter("items", items);
    env.add_filter("length", length);
    env.add_filter("reverse", reverse);
    env.add_filter("safe", safe);
    env.add_filter("string", string);
    env.add_filter("tojson", tojson);
    env.add_filter("trim", |value: Value| trim(&value));

    env.add_test("defined", |v: Value| !v.is_undefined());
    env.add_test("undefined", |v: Value| v.is_undefined());
    env.add_test("none", |v: Value| v.is_none());
    env.add_test("string", |v: Value| v.kind() == ValueKind::String);
    env.add_test("mapping", |v: Value| v.kind() == ValueKind::Map);
    // llama.cpp calls undefined values and mappings iterable and sequences too.
    env.add_test("iterable", is_sequence_like);
    env.add_test("sequence", is_sequence_like);
    env.add_test("true", |v: Value| v.kind() == ValueKind::Bool && v.is_true());
    env.add_test("false", |v: Value| v.kind() == ValueKind::Bool && !v.is_true());
    env
}

fn is_sequence_like(v: Value) -> bool {
    matches!(v.kind(), ValueKind::Map | ValueKind::Seq | ValueKind::String | ValueKind::Undefined)
}

/// The error for something the dialect leaves out, found while the template runs.
fn unsupported(what: String) -> Error {
    Error::new(ErrorKind::InvalidOperation, format!("{what} is not supported"))
}

fn raise_exception(message: Value) -> Result<Value, Error> {
    match message.as_str() {
        Some(message) => {
            Err(Error::new(ErrorKind::InvalidOperation, format!("Jinja Exception: {message}")))
        }
        None => Err(unsupported("raise_exception with something other than a string".into())),
    }
}

/// How an expression `{{ ... }}` is written into the prompt: strings and
/// numbers as they are, booleans as Python writes them, a list as its items one
/// after the other; none, undefined and mappings write nothing.
fn format_value(out: &mut minijinja::Output, _state: &State, value: &Value) -> Result<(), Error> {
    fn write(out: &mut String, value: &Value) -> Result<(), Error> {
        match value.kind() {
            ValueKind::Undefined | ValueKind::None | ValueKind::Map => {}
            ValueKind::Bool => out.push_str(if value.is_true() { "True" } else { "False" }),
            ValueKind::Number => out.push_str(&number_text(value)?),
            ValueKind::String => out.push_str(value.as_str().unwrap_or_default()),
            ValueKind::Seq => {
                for item in value.try_iter()? {
                    write(out, &item)?;
                }
            }
            _ => return Err(unsupported(format!("writing a {} into the prompt", value.kind()))),
        }
        Ok(())
    }
    let mut text = String::new();
    write(&mut text, value)?;
    std::fmt::Write::write_str(out, &text).map_err(|_| Error::from(ErrorKind::WriteFailure))
}

/// A number as llama.cpp's `{{ n }}` writes it. Integers in decimal; floats with
/// six decimals and the trailing zeros cut, keeping one.
fn number_text(value: &Value) -> Result<String, Error> {
    if value.is_integer() {
        return Ok(value.to_string());
    }
    let f =
        f64::try_from(value.clone()).map_err(|_| unsupported("a number out of range".into()))?;
    let mut text = format!("{f:.6}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.push('0');
    }
    Ok(text)
}

/// C's `isspace` in the C locale, which is what llama.cpp strips and splits by.
fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

fn strip_bytes(text: &str, left: bool, right: bool, chars: Option<&str>) -> Result<String, Error> {
    let matches = |b: u8| match chars {
        Some(chars) => chars.as_bytes().contains(&b),
        None => is_space(b),
    };
    if chars.is_some_and(|c| !c.is_ascii()) {
        return Err(unsupported("strip with characters outside ASCII".into()));
    }
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut end = bytes.len();
    if left {
        while start < end && matches(bytes[start]) {
            start += 1;
        }
    }
    if right {
        while end > start && matches(bytes[end - 1]) {
            end -= 1;
        }
    }
    // Only ASCII bytes are cut, so the rest is whole characters.
    Ok(text[start..end].to_owned())
}

/// `|trim`. Other values than strings are turned into text first, the way
/// llama.cpp writes them in a template (a float with six decimals, cut).
fn trim(value: &Value) -> Result<Value, Error> {
    let text = match value.kind() {
        ValueKind::String | ValueKind::Undefined | ValueKind::Bool | ValueKind::None => {
            python_string(value)?
        }
        ValueKind::Number => number_text(value)?,
        kind => return Err(unsupported(format!("the filter trim on a {kind}"))),
    };
    strip_bytes(&text, true, true, None).map(Value::from)
}

/// `str(value)` for the values llama.cpp writes the way Python does.
fn python_string(value: &Value) -> Result<String, Error> {
    match value.kind() {
        ValueKind::String => Ok(value.as_str().unwrap_or_default().to_owned()),
        ValueKind::Undefined => Ok(String::new()),
        ValueKind::None => Ok("None".into()),
        ValueKind::Bool => Ok(if value.is_true() { "True" } else { "False" }.into()),
        ValueKind::Number => {
            if value.is_integer() {
                Ok(value.to_string())
            } else {
                Ok(float_repr(
                    f64::try_from(value.clone())
                        .map_err(|_| unsupported("a number out of range".into()))?,
                ))
            }
        }
        kind => Err(unsupported(format!("turning a {kind} into a string"))),
    }
}

fn string(value: Value) -> Result<Value, Error> {
    python_string(&value).map(Value::from)
}

fn safe(value: Value) -> Result<Value, Error> {
    match value.kind() {
        ValueKind::String => Ok(value),
        _ => string(value),
    }
}

/// `|default(fallback, boolean=false)`: the fallback replaces an undefined or
/// none value, or with `boolean` any false one. llama.cpp has no `default` for
/// mappings.
fn default(
    value: Value,
    fallback: Value,
    boolean: Option<Value>,
    kwargs: Kwargs,
) -> Result<Value, Error> {
    let boolean: Option<Value> = boolean.or(kwargs.get("boolean")?);
    kwargs.assert_all_used()?;
    if value.kind() == ValueKind::Map {
        return Err(unsupported("the filter default on a mapping".into()));
    }
    let missing = if boolean.is_some_and(|b| b.is_true()) {
        !value.is_true()
    } else {
        value.is_undefined() || value.is_none()
    };
    Ok(if missing { fallback } else { value })
}

fn length(value: Value) -> Result<usize, Error> {
    match value.kind() {
        ValueKind::Undefined => Ok(0),
        // llama.cpp counts the bytes of a string.
        ValueKind::String => Ok(value.as_str().unwrap_or_default().len()),
        ValueKind::Seq | ValueKind::Map => Ok(value.len().unwrap_or(0)),
        kind => Err(unsupported(format!("the filter length on a {kind}"))),
    }
}

/// The items of a list from the last to the first. Templates write this as
/// `list[::-1]`, which [`prepare`] turns into `list|reverse`.
fn reverse(value: Value) -> Result<Vec<Value>, Error> {
    match value.kind() {
        ValueKind::Seq => {
            let mut items: Vec<Value> = value.try_iter()?.collect();
            items.reverse();
            Ok(items)
        }
        ValueKind::Undefined => Ok(Vec::new()),
        kind => Err(unsupported(format!("[::-1] on a {kind}"))),
    }
}

fn items(value: Value) -> Result<Vec<Value>, Error> {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => Ok(Vec::new()),
        ValueKind::Map => value
            .try_iter()?
            .map(|key| {
                let item = value.get_item(&key)?;
                Ok(Value::from(vec![key, item]))
            })
            .collect(),
        kind => Err(unsupported(format!("the filter items on a {kind}"))),
    }
}

fn tojson(value: Value, rest: Rest<Value>) -> Result<String, Error> {
    if !rest.is_empty() {
        return Err(unsupported("tojson with arguments".into()));
    }
    if value.is_undefined() {
        return Err(unsupported("the filter tojson on an undefined value".into()));
    }
    let mut out = String::new();
    json_text(&mut out, &value)?;
    Ok(out)
}

/// The value as `json.dumps` writes it with its defaults: ", " between items,
/// ": " between a key and its value, non-ASCII text as it is.
pub(crate) fn json_text(out: &mut String, value: &Value) -> Result<(), Error> {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if value.is_integer() {
                let _ = write!(out, "{value}");
            } else {
                out.push_str(&float_json(
                    f64::try_from(value.clone())
                        .map_err(|_| unsupported("a number out of range".into()))?,
                ));
            }
        }
        ValueKind::String => write_string(out, value.as_str().unwrap_or_default()),
        ValueKind::Seq => {
            out.push('[');
            for (i, item) in value.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                json_text(out, &item)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, key) in value.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let Some(name) = key.as_str() else {
                    return Err(unsupported(
                        "a mapping with keys that are not strings in tojson".into(),
                    ));
                };
                write_string(out, name);
                out.push_str(": ");
                json_text(out, &value.get_item(&key)?)?;
            }
            out.push('}');
        }
        kind => return Err(unsupported(format!("tojson of a {kind}"))),
    }
    Ok(())
}

/// The methods of strings and mappings.
fn call_method(_state: &State, value: &Value, name: &str, args: &[Value]) -> Result<Value, Error> {
    if !METHODS.contains(&name) {
        return Err(Error::from(ErrorKind::UnknownMethod));
    }
    let kwargs = args.last().and_then(|v| Kwargs::try_from(v.clone()).ok());
    if kwargs.is_some() {
        return Err(unsupported(format!("keyword arguments to .{name}()")));
    }
    let text_arg = |i: usize| -> Result<Option<String>, Error> {
        match args.get(i) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(|s| Some(s.to_owned()))
                .ok_or_else(|| unsupported(format!(".{name}() takes strings"))),
        }
    };
    match (value.kind(), name) {
        (ValueKind::String, "startswith" | "endswith") => {
            let (Some(text), Some(affix), 1) = (value.as_str(), text_arg(0)?, args.len()) else {
                return Err(unsupported(format!(".{name}() takes one string")));
            };
            Ok(Value::from(if name == "startswith" {
                text.starts_with(&affix)
            } else {
                text.ends_with(&affix)
            }))
        }
        (ValueKind::String, "strip" | "lstrip" | "rstrip") => {
            if args.len() > 1 {
                return Err(unsupported(format!(".{name}() takes at most one argument")));
            }
            let text = value.as_str().unwrap_or_default();
            let chars = text_arg(0)?;
            strip_bytes(text, name != "rstrip", name != "lstrip", chars.as_deref()).map(Value::from)
        }
        (ValueKind::String, "split" | "rsplit") => {
            split(value.as_str().unwrap_or_default(), name == "rsplit", args)
        }
        (ValueKind::Map, "items") if args.is_empty() => items(value.clone()).map(Value::from),
        (ValueKind::Map, "keys") if args.is_empty() => {
            Ok(Value::from(value.try_iter()?.collect::<Vec<_>>()))
        }
        (ValueKind::Map, "values") if args.is_empty() => {
            let values: Result<Vec<_>, _> = value.try_iter()?.map(|k| value.get_item(&k)).collect();
            Ok(Value::from(values?))
        }
        (ValueKind::Map, "get") if (1..=2).contains(&args.len()) => {
            if args[0].kind() != ValueKind::String {
                return Err(unsupported(".get() takes a string key".into()));
            }
            let found = value.get_item(&args[0])?;
            Ok(if found.is_undefined() {
                args.get(1).cloned().unwrap_or(Value::from(()))
            } else {
                found
            })
        }
        _ => Err(unsupported(format!(
            "the method .{name}() on a {} or with these arguments",
            value.kind()
        ))),
    }
}

/// `str.split(delimiter, maxsplit)` as llama.cpp does it: the delimiter is a
/// string, a space by default, never a run of white space.
fn split(text: &str, from_right: bool, args: &[Value]) -> Result<Value, Error> {
    if args.len() > 2 {
        return Err(unsupported("split takes a delimiter and a limit".into()));
    }
    let delimiter = match args.first() {
        None => " ".to_owned(),
        Some(d) => d.as_str().ok_or_else(|| unsupported("split takes a string".into()))?.to_owned(),
    };
    if delimiter.is_empty() {
        return Err(unsupported("empty separator".into()));
    }
    let mut limit = match args.get(1) {
        None => -1,
        Some(n) => n.as_i64().ok_or_else(|| unsupported("split takes a number as limit".into()))?,
    };
    let mut parts = Vec::new();
    let mut rest = text;
    if from_right {
        while limit != 0 {
            let Some(at) = rest.rfind(&delimiter) else { break };
            parts.push(Value::from(rest[at + delimiter.len()..].to_owned()));
            rest = &rest[..at];
            limit -= 1;
        }
        parts.push(Value::from(rest.to_owned()));
        parts.reverse();
    } else {
        while limit != 0 {
            let Some(at) = rest.find(&delimiter) else { break };
            parts.push(Value::from(rest[..at].to_owned()));
            rest = &rest[at + delimiter.len()..];
            limit -= 1;
        }
        parts.push(Value::from(rest.to_owned()));
    }
    Ok(Value::from(parts))
}
