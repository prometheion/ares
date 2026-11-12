mod common;

use ares_grammar::{Error, Source};
use common::*;
use serde_json::{Value, json};

/// Walks the schema many times and returns the texts of the walks that ended,
/// parsed.
fn samples(toy: &Toy, schema: &Value, walks: u64) -> Vec<Value> {
    let mut g = toy.grammar(Source::JsonSchema(&schema.to_string()));
    let mut out = Vec::new();
    for seed in 1..=walks {
        let (tokens, ended) = walk(toy, &mut g, seed, 500);
        if ended {
            let text = toy.text(&tokens);
            match serde_json::from_str(&text) {
                Ok(value) => out.push(value),
                // A number with an exponent past what a double holds is still JSON.
                Err(e) if e.to_string().contains("out of range") => {}
                Err(e) => panic!("{text:?}: {e}"),
            }
        }
    }
    assert!(out.len() as u64 > walks / 4, "only {} of {walks} walks ended", out.len());
    out
}

fn accepts_json(toy: &Toy, schema: &Value, text: &str) -> bool {
    let mut g = toy.grammar(Source::JsonSchema(&schema.to_string()));
    accepts(toy, &mut g, text)
}

#[test]
fn an_object_schema_gives_objects_with_the_required_members() {
    let toy = toy();
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 12},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "kind": {"enum": ["cat", "dog"]},
            "tags": {"type": "array", "items": {"type": "boolean"}, "maxItems": 3},
        },
        "required": ["name", "age", "kind"],
        "additionalProperties": false,
    });
    for value in samples(&toy, &schema, 60) {
        let object = value.as_object().expect("an object");
        assert!(object.keys().all(|k| ["name", "age", "kind", "tags"].contains(&k.as_str())));
        assert!(object["name"].as_str().unwrap().chars().count() <= 12, "{value}");
        let age = object["age"].as_i64().expect("an integer");
        assert!((0..=150).contains(&age), "{value}");
        assert!(["cat", "dog"].contains(&object["kind"].as_str().unwrap()), "{value}");
        if let Some(tags) = object.get("tags") {
            assert!(
                tags.as_array().unwrap().len() <= 3
                    && tags.as_array().unwrap().iter().all(Value::is_boolean)
            );
        }
    }
}

#[test]
fn a_schema_refuses_text_that_does_not_fit_it() {
    let toy = toy();
    let schema = json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
        "required": ["n"],
        "additionalProperties": false,
    });
    assert!(accepts_json(&toy, &schema, r#"{"n":1}"#));
    assert!(accepts_json(&toy, &schema, r#"{"n": 1, "s": "x"}"#));
    assert!(!accepts_json(&toy, &schema, r#"{"s":"x"}"#), "n is required");
    assert!(!accepts_json(&toy, &schema, r#"{"n":"1"}"#), "n is an integer");
    assert!(!accepts_json(&toy, &schema, r#"{"n":1,"z":2}"#), "no other members");
    assert!(!accepts_json(&toy, &schema, r#"[1]"#));
    assert!(!accepts_json(&toy, &schema, r#"{"n":1.5}"#));
}

#[test]
fn references_alternatives_and_patterns_are_followed() {
    let toy = toy();
    let schema = json!({
        "$defs": {"id": {"type": "string", "pattern": "^[a-f0-9]{4}$"}},
        "type": "object",
        "properties": {
            "id": {"$ref": "#/$defs/id"},
            "v": {"anyOf": [{"type": "integer"}, {"type": "null"}, {"const": "x"}]},
        },
        "required": ["id", "v"],
        "additionalProperties": false,
    });
    for value in samples(&toy, &schema, 60) {
        let id = value["id"].as_str().unwrap();
        assert!(
            id.len() == 4 && id.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "{value}"
        );
        let v = &value["v"];
        // An integer past 64 bits reads as a double here.
        assert!(v.as_f64().is_some_and(|f| f.fract() == 0.0) || v.is_null() || v == "x", "{value}");
    }
}

#[test]
fn the_json_object_schema_gives_objects_only() {
    let toy = toy();
    let schema = json!({"type": "object"});
    assert!(accepts_json(&toy, &schema, r#"{}"#));
    assert!(accepts_json(&toy, &schema, r#"{"a":[1,{"b":null}],"c":"d"}"#));
    assert!(!accepts_json(&toy, &schema, r#"[1]"#));
    assert!(!accepts_json(&toy, &schema, r#"1"#));
    for value in samples(&toy, &schema, 40) {
        assert!(value.is_object());
    }
}

#[test]
fn white_space_between_the_parts_is_bounded_like_llama_cpps() {
    let toy = toy();
    let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": false});
    assert!(accepts_json(&toy, &schema, "{\"a\":1}"));
    assert!(accepts_json(&toy, &schema, "{ \"a\": 1 }"));
    assert!(accepts_json(&toy, &schema, "{\n\n\t\t\"a\":\n  1\n}"));
    assert!(!accepts_json(&toy, &schema, "{  \"a\":1}"), "two spaces in a row");
    assert!(!accepts_json(&toy, &schema, "{\n\n\n\"a\":1}"), "three line breaks");
}

#[test]
fn keywords_the_engine_does_not_know_are_left_out_as_llama_cpp_leaves_them() {
    let toy = toy();
    let schema = json!({"type": "string", "x-note": "ignored", "format": "no-such-format"});
    assert!(accepts_json(&toy, &schema, r#""hello""#));
    let schema = json!(true);
    assert!(accepts_json(&toy, &schema, r#"{"any":[1,2]}"#));
}

#[test]
fn a_schema_that_is_wrong_is_refused_with_the_reason() {
    let toy = toy();
    let message = |source: Source<'_>| match toy.compiler.compile(source) {
        Ok(_) => panic!("{source:?} should not compile"),
        Err(e) => e,
    };
    assert!(
        matches!(message(Source::JsonSchema("{")), Error::Schema(m) if m.contains("not valid JSON"))
    );
    assert!(
        matches!(message(Source::JsonSchema("[1]")), Error::Schema(m) if m.contains("an object or a boolean"))
    );
    let remote =
        json!({"type": "object", "properties": {"a": {"$ref": "http://example.invalid/x.json"}}});
    let e = message(Source::JsonSchema(&remote.to_string()));
    assert!(matches!(&e, Error::Compile(m) if m.contains("JSON schema")), "{e:?}");
    let bad_type = json!({"type": "nonsense"});
    let e = message(Source::JsonSchema(&bad_type.to_string()));
    assert!(matches!(&e, Error::Compile(_)), "{e:?}");
    let unsatisfiable = json!({"type": "integer", "minimum": 5, "maximum": 1});
    let e = message(Source::JsonSchema(&unsatisfiable.to_string()));
    assert!(matches!(&e, Error::Compile(_)), "{e:?}");
}
