//! The defaults of a model's requests: every option, and which value holds
//! where the request, the model's section, `[*]` and the command line say
//! different things.

use std::path::Path;

use ares_api::{RequestId, SamplerKind, Sampling};
use serde_json::{Value, json};

use super::*;
use crate::options::{self, Kind, ModelOptions};
use crate::preset::{Preset, Presets};
use crate::wire::{self, Endpoint};

/// Every setting with three values its option takes, all different. A
/// number that may have a fraction is written with one, as the listing
/// writes it.
const VALUES: &[(&str, [&str; 3])] = &[
    ("--temperature", ["0.5", "0.7", "1.25"]),
    ("--top-k", ["20", "40", "0"]),
    ("--top-p", ["0.8", "0.95", "1.0"]),
    ("--min-p", ["0.05", "0.1", "0.0"]),
    ("--typical-p", ["0.9", "0.5", "1.0"]),
    ("--top-n-sigma", ["1.5", "2.0", "0.0"]),
    ("--xtc-probability", ["0.5", "0.25", "1.0"]),
    ("--xtc-threshold", ["0.1", "0.2", "0.5"]),
    ("--dynatemp-range", ["0.5", "0.25", "0.0"]),
    ("--dynatemp-exponent", ["1.0", "1.5", "2.0"]),
    ("--repeat-penalty", ["1.1", "1.05", "1.0"]),
    ("--presence-penalty", ["1.5", "-0.5", "0.0"]),
    ("--frequency-penalty", ["0.5", "-2.0", "2.0"]),
    ("--repeat-last-n", ["64", "256", "0"]),
    ("--dry-multiplier", ["0.8", "0.5", "0.0"]),
    ("--dry-base", ["1.75", "2.0", "1.5"]),
    ("--dry-allowed-length", ["2", "3", "4"]),
    ("--dry-penalty-last-n", ["64", "1024", "0"]),
    ("--dry-sequence-breakers", [r#"["\n", ":"]"#, r#"["--", "a, b"]"#, "[]"]),
    ("--mirostat", ["2", "1", "0"]),
    ("--mirostat-tau", ["5.0", "4.5", "3.0"]),
    ("--mirostat-eta", ["0.1", "0.2", "0.0"]),
    ("--samplers", ["top_k,temperature", "penalties,dry,min_p,temperature", "temperature"]),
    ("--seed", ["42", "7", "0"]),
];

fn setting(flag: &str) -> &'static Setting {
    SETTINGS.iter().find(|setting| setting.flag == flag).expect("a setting")
}

/// The value of an option as a request writes it in its field.
fn in_a_request(flag: &str, text: &str) -> Value {
    if flag == "--samplers" {
        return text.split(',').collect::<Vec<_>>().into();
    }
    serde_json::from_str(text).expect("the value of an option is JSON")
}

fn with(options: &[(&str, &str)]) -> Defaults {
    let mut model = ModelOptions::default();
    for (name, value) in options {
        let (flag, kind) = options::lookup(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(kind, Kind::Value);
        options::set(&mut model, flag, value).unwrap_or_else(|e| panic!("{e}"));
    }
    model.defaults
}

/// A chat request with `members` next to its model and its message, as the
/// reader of bodies understands it.
fn request(members: Value) -> wire::Parsed {
    let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    body.as_object_mut().unwrap().extend(members.as_object().unwrap().clone());
    wire::parse(Endpoint::Chat, body.to_string().as_bytes(), RequestId(1)).expect("a request")
}

fn shown(setting: &Setting, sampling: &Sampling) -> Option<Value> {
    (setting.show)(sampling)
}

#[test]
fn every_setting_is_an_option_with_the_name_of_the_field_of_a_request() {
    assert_eq!(
        VALUES.iter().map(|(flag, _)| *flag).collect::<Vec<_>>(),
        SETTINGS.iter().map(|setting| setting.flag).collect::<Vec<_>>(),
        "the table of this test has every setting"
    );
    for setting in SETTINGS {
        assert_eq!(setting.flag, format!("--{}", setting.field.replace('_', "-")));
        assert_eq!(options::lookup(setting.flag), Some((setting.flag, Kind::Value)));
        assert!(!options::repeats(setting.flag));
    }
    for flag in ["--max-tokens", "--max-tokens-limit"] {
        assert_eq!(options::lookup(flag), Some((flag, Kind::Value)));
    }
    // The names of llama-server that Ares writes differently are not options.
    for name in ["--temp", "--typical", "--n-predict", "--mirostat-lr", "--sampler-seq", "-s"] {
        assert_eq!(options::lookup(name), None, "{name}");
    }
    assert_eq!(STAGES_MAX, ares_sampler::MAX_STAGES, "the engine's limit");
}

#[test]
fn an_option_sets_the_default_of_its_field_and_of_no_other() {
    for (flag, values) in VALUES {
        let setting = setting(flag);
        for value in values {
            let defaults = with(&[(flag, value)]);
            let expected = in_a_request(flag, value);
            assert_eq!(shown(setting, &defaults.sampling), Some(expected.clone()), "{flag}");
            // What the listing says is what a request would write.
            assert_eq!(defaults.fields(), json!({ setting.field: expected }), "{flag} {value}");
            assert_eq!(defaults.sampling.validate(), Ok(()), "{flag} {value}");
        }
    }
    assert_eq!(Defaults::default().fields(), json!({}));
}

#[test]
fn the_value_of_a_request_holds_over_the_model_s_and_that_over_the_engine_s() {
    for (flag, [own, model, _]) in VALUES {
        let setting = setting(flag);
        let defaults = with(&[(flag, model)]);
        let (own, model) = (in_a_request(flag, own), in_a_request(flag, model));

        // The request says it: its value, whatever the model has.
        let mut parsed = request(json!({ setting.field: own }));
        defaults.apply(&mut parsed);
        assert_eq!(shown(setting, &parsed.request.sampling), Some(own.clone()), "{flag}");

        // It leaves the field out, or writes null: the model's.
        for members in [json!({}), json!({ setting.field: null })] {
            let mut parsed = request(members);
            defaults.apply(&mut parsed);
            assert_eq!(shown(setting, &parsed.request.sampling), Some(model.clone()), "{flag}");
            // And no other field was given a value.
            let mut one = Sampling::default();
            (setting.fill)(&mut one, &defaults.sampling);
            assert_eq!(parsed.request.sampling, one, "{flag}");
        }

        // The model has no default: the field stays open for the engine.
        let mut parsed = request(json!({}));
        Defaults::default().apply(&mut parsed);
        assert_eq!(parsed.request.sampling, Sampling::default(), "{flag}");
        let mut parsed = request(json!({ setting.field: own }));
        Defaults::default().apply(&mut parsed);
        assert_eq!(shown(setting, &parsed.request.sampling), Some(own), "{flag}");
    }
}

fn presets(text: &str, given: &[(&str, &str)]) -> Vec<Preset> {
    let given: Vec<_> = given
        .iter()
        .map(|(name, value)| (options::lookup(name).unwrap().0, (*value).to_owned()))
        .collect();
    Presets::parse(text, Path::new("m.ini"), &given, &|_| true)
        .unwrap_or_else(|e| panic!("{e}\n{text}"))
        .models
}

#[test]
fn the_command_line_holds_over_a_section_and_a_section_over_the_global_one() {
    let lengths =
        [("--max-tokens", ["100", "200", "300"]), ("--max-tokens-limit", ["1", "2", "3"])];
    for (flag, [global, section, given]) in VALUES.iter().chain(&lengths) {
        let key = flag.trim_start_matches('-');
        let text = format!(
            "[*]\n{key} = {global}\n[own]\nmodel = /a.gguf\n{key} = {section}\n\
             [global]\nmodel = /b.gguf\n"
        );
        let value = |preset: &Preset| {
            let defaults = &preset.options.defaults;
            match *flag {
                "--max-tokens" => defaults.max_tokens.map(Value::from),
                "--max-tokens-limit" => defaults.max_tokens_limit.map(Value::from),
                flag => shown(setting(flag), &defaults.sampling),
            }
        };
        let said = |text: &str| Some(in_a_request(flag, text));

        let models = presets(&text, &[]);
        let [own, from_global] = &models[..] else { panic!() };
        assert_eq!(value(own), said(section), "{flag}: the section over [*]");
        assert_eq!(value(from_global), said(global), "{flag}: [*] over the built-in");

        let models = presets(&text, &[(*flag, *given)]);
        let [own, from_global] = &models[..] else { panic!() };
        assert_eq!(value(own), said(given), "{flag}: the command line over the section");
        assert_eq!(value(from_global), said(given), "{flag}: the command line over [*]");

        // Without the key anywhere the model has no default, and the engine's holds.
        let models = presets("[plain]\nmodel = /a.gguf\n", &[]);
        assert_eq!(value(&models[0]), None, "{flag}");
        assert_eq!(models[0].options.defaults, Defaults::default());
        // On the command line of one model it is that model's.
        let mut single = ModelOptions::default();
        options::set(&mut single, "--model", "/a.gguf").unwrap();
        options::set(&mut single, options::lookup(flag).unwrap().0, given).unwrap();
        let models = Presets::single(single).models;
        assert_eq!(value(&models[0]), said(given), "{flag}");
    }
}

#[test]
fn a_value_outside_the_range_of_its_field_is_refused_by_the_option() {
    for (flag, bad) in [
        ("--temperature", &["-0.1", "nan", "inf", "hot", ""][..]),
        ("--top-k", &["-1", "1.5", "many"]),
        ("--top-p", &["0", "1.01", "-0.5"]),
        ("--min-p", &["-0.1", "1.5"]),
        ("--typical-p", &["0", "2"]),
        ("--top-n-sigma", &["inf", "wide"]),
        ("--xtc-probability", &["-0.1", "1.1"]),
        ("--xtc-threshold", &["-0.1", "1.1"]),
        ("--dynatemp-range", &["-1", "inf"]),
        ("--dynatemp-exponent", &["-1"]),
        ("--repeat-penalty", &["0", "-1", "inf"]),
        ("--presence-penalty", &["2.5", "-2.5"]),
        ("--frequency-penalty", &["2.5", "-2.5"]),
        ("--repeat-last-n", &["-1", "1025", "all"]),
        ("--dry-multiplier", &["-1"]),
        ("--dry-base", &["-1"]),
        ("--dry-allowed-length", &["-1", "1025"]),
        ("--dry-penalty-last-n", &["-1", "1025"]),
        ("--dry-sequence-breakers", &["\\n", "[1, 2]", r#"{"a": "b"}"#, r#""\n""#, "none"]),
        ("--mirostat", &["3", "-1", "on"]),
        ("--mirostat-tau", &["0", "-1"]),
        ("--mirostat-eta", &["-0.1"]),
        (
            "--samplers",
            &[
                "",
                "top_k;temperature",
                "top_k,penalties",
                "penalties,penalties,top_k",
                "dry,dry",
                "top_k,nothing",
                "infill",
                // Seventeen stages that cut.
                "top_k,top_p,min_p,top_k,top_p,min_p,top_k,top_p,min_p,top_k,top_p,min_p,top_k,\
                 top_p,min_p,top_k,top_p",
            ],
        ),
        ("--seed", &["-1", "1.5", "random"]),
        ("--max-tokens", &["0", "-1", "all", "1.5"]),
        ("--max-tokens-limit", &["0", "-1", "none"]),
    ] {
        for value in bad {
            let mut model = ModelOptions::default();
            let error = options::set(&mut model, options::lookup(flag).unwrap().0, value)
                .expect_err(&format!("{flag} {value}"));
            assert_eq!((error.flag, error.value.as_str()), (flag, *value));
            assert_eq!(model, ModelOptions::default(), "{flag} {value}: nothing is set");
            if let Some(setting) = SETTINGS.iter().find(|setting| setting.flag == flag) {
                assert_eq!(error.expected, setting.expected);
            }
        }
    }
    let said = options::set(&mut ModelOptions::default(), "--top-p", "1.5").unwrap_err();
    assert_eq!(
        said.to_string(),
        "option '--top-p': '1.5' is not a number above 0 and at most 1 (1 cuts nothing)"
    );
    // A whole number is a number, and white space around a value is not part of it.
    let read = with(&[("--top-p", "1"), ("--temperature", " 2 "), ("--top-k", " 40")]).sampling;
    assert_eq!((read.top_p, read.temperature, read.top_k), (Some(1.0), Some(2.0), Some(40)));
    // Sixteen stages that cut are a program; penalties and dry stand before them.
    let sixteen = format!("dry,penalties,{}", ["top_k"; 16].join(","));
    assert_eq!(with(&[("--samplers", sixteen.as_str())]).sampling.samplers.unwrap().len(), 18);
    let named = with(&[("--samplers", "Top-K, typical, temp")]).sampling.samplers.unwrap();
    assert_eq!(named, [SamplerKind::TopK, SamplerKind::TypicalP, SamplerKind::Temperature]);
}

#[test]
fn a_reply_is_as_long_as_the_request_says_within_the_most_the_model_allows() {
    let length = |options: &[(&str, &str)], asked| with(options).length(asked);
    // Nothing set: what the request says, and no limit when it names none.
    assert_eq!(length(&[], None), (None, false));
    assert_eq!(length(&[], Some(1_000_000)), (Some(1_000_000), false));
    // A default: for the request that names no limit, and no bound on one that does.
    let default = [("--max-tokens", "4096")];
    assert_eq!(length(&default, None), (Some(4096), false));
    assert_eq!(length(&default, Some(16)), (Some(16), false));
    assert_eq!(length(&default, Some(100_000)), (Some(100_000), false));
    // A limit: no request gets more, and one that names none gets it.
    let limit = [("--max-tokens-limit", "8192")];
    assert_eq!(length(&limit, None), (Some(8192), false));
    assert_eq!(length(&limit, Some(8192)), (Some(8192), false));
    assert_eq!(length(&limit, Some(8193)), (Some(8192), true));
    assert_eq!(length(&limit, Some(u32::MAX)), (Some(8192), true));
    assert_eq!(length(&limit, Some(16)), (Some(16), false));
    // Both.
    let both = [("--max-tokens", "4096"), ("--max-tokens-limit", "8192")];
    assert_eq!(length(&both, None), (Some(4096), false));
    assert_eq!(length(&both, Some(6000)), (Some(6000), false));
    assert_eq!(length(&both, Some(9000)), (Some(8192), true));

    // A request says its limit under any of its names, and -1 is none.
    for (members, expected, lowered) in [
        (json!({}), Some(4096), None),
        (json!({"max_tokens": null}), Some(4096), None),
        (json!({"max_tokens": -1}), Some(4096), None),
        (json!({"n_predict": -1}), Some(4096), None),
        (json!({"max_tokens": 5000}), Some(5000), None),
        (json!({"max_completion_tokens": 5000}), Some(5000), None),
        (json!({"n_predict": 9000}), Some(8192), Some(8192)),
        (json!({"max_tokens": 9000}), Some(8192), Some(8192)),
        (json!({"max_completion_tokens": 8193}), Some(8192), Some(8192)),
    ] {
        let mut parsed = request(members.clone());
        with(&both).apply(&mut parsed);
        assert_eq!((parsed.request.max_tokens, parsed.lowered), (expected, lowered), "{members}");
    }

    // The two hold together only if the default is within the limit.
    assert_eq!(with(&both).check(), Ok(()));
    assert_eq!(with(&[("--max-tokens", "8192"), ("--max-tokens-limit", "8192")]).check(), Ok(()));
    let conflict =
        with(&[("--max-tokens-limit", "4096"), ("--max-tokens", "8192")]).check().unwrap_err();
    assert_eq!(conflict.flags, ["--max-tokens", "--max-tokens-limit"]);
    assert!(
        conflict.reason.starts_with("max-tokens = 8192 is more than max-tokens-limit = 4096"),
        "{conflict}"
    );
}

#[test]
fn a_request_that_asks_for_a_seed_of_its_own_draw_does_not_get_the_model_s() {
    let defaults = with(&[("--seed", "42")]);
    for (members, seed) in [
        (json!({}), Some(42)),
        (json!({"seed": null}), Some(42)),
        (json!({"seed": 7}), Some(7)),
        (json!({"seed": 0}), Some(0)),
        // llama.cpp's "pick a random seed".
        (json!({"seed": -1}), None),
        (json!({"seed": -1.0}), None),
    ] {
        let mut parsed = request(members.clone());
        defaults.apply(&mut parsed);
        assert_eq!(parsed.request.sampling.seed, seed, "{members}");
    }
    // Without a seed of the model nothing changes.
    let mut parsed = request(json!({"seed": -1}));
    Defaults::default().apply(&mut parsed);
    assert_eq!(parsed.request.sampling.seed, None);
}

#[test]
fn what_is_not_a_default_is_left_as_the_request_has_it() {
    // A model's options cannot set a bias, a flag or the probabilities to
    // report; a `Defaults` built by hand that has them does not apply them.
    let mut defaults = with(&[("--temperature", "0.7")]);
    defaults.sampling.ignore_eos = true;
    defaults.sampling.logprobs = Some(ares_api::LogprobRequest { top: 3, after_chain: false });
    let mut parsed = request(json!({}));
    defaults.apply(&mut parsed);
    let expected = Sampling { temperature: Some(0.7), ..Sampling::default() };
    assert_eq!(parsed.request.sampling, expected);
}

#[test]
fn the_listing_and_the_log_name_the_defaults_as_a_request_writes_them() {
    let defaults = with(&[
        ("--temperature", "1.0"),
        ("--top-p", "0.95"),
        ("--top-k", "20"),
        ("--min-p", "0.0"),
        ("--presence-penalty", "0.0"),
        ("--repeat-penalty", "1.0"),
        ("--max-tokens", "96000"),
    ]);
    assert_eq!(
        defaults.fields(),
        json!({
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repeat_penalty": 1.0, "max_tokens": 96000
        })
    );
    // Written as the number that was set, not as its nearest f32 in 17 digits.
    assert_eq!(defaults.fields()["top_p"].to_string(), "0.95");
    assert_eq!(
        defaults.to_string(),
        "a field a request leaves out: temperature=1.0 top_k=20 top_p=0.95 min_p=0.0 \
         repeat_penalty=1.0 presence_penalty=0.0 max_tokens=96000"
    );

    assert_eq!(Defaults::default().to_string(), "a field a request leaves out is the engine's own");
    // A limit alone is also what a request gets that names none.
    let limited = with(&[("--max-tokens-limit", "4096"), ("--samplers", "top_k,temp")]);
    assert_eq!(limited.fields(), json!({"samplers": ["top_k", "temperature"], "max_tokens": 4096}));
    assert_eq!(
        limited.to_string(),
        "a field a request leaves out: samplers=[\"top_k\",\"temperature\"] max_tokens=4096; \
         no reply is longer than 4096 tokens"
    );
    let only_limit = with(&[("--max-tokens", "100"), ("--max-tokens-limit", "200")]);
    assert_eq!(only_limit.fields(), json!({"max_tokens": 100}));
}
