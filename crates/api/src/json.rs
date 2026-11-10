//! A JSON value that does not depend on a JSON library.

use std::fmt;

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A JSON value, kept as the client sent it.
///
/// Tool definitions and tool calls are read by the model's chat template, not
/// by this engine, so they must survive the trip unchanged. That rules out
/// `serde_json::Value` (this crate has no JSON dependency) and rules out a
/// map type that sorts its keys: a template that renders a tool schema shows
/// the properties in the order the client wrote them, and a model that was
/// trained on that order should see it. So objects are a list of pairs in
/// document order, duplicate keys included.
///
/// The value is exact, the spelling is not: members, order, strings and
/// integers come back as they went in, but `1e2` is read as the float `100.0`
/// and an integer that does not fit 64 bits becomes a float, as in most JSON
/// parsers. Integers have exactly one representation: [`Json::Uint`] exists
/// only for values above `i64::MAX`.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    /// An integer above `i64::MAX`.
    Uint(u64),
    Float(f64),
    String(String),
    Array(Vec<Json>),
    /// Members in document order.
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(members) => Some(members),
            _ => None,
        }
    }

    /// The member `key` of an object. When the key appears more than once the
    /// last occurrence wins, as it does in common JSON parsers.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A short name of the kind of value, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Int(_) | Json::Uint(_) | Json::Float(_) => "a number",
            Json::String(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => serializer.serialize_unit(),
            Json::Bool(b) => serializer.serialize_bool(*b),
            Json::Int(i) => serializer.serialize_i64(*i),
            Json::Uint(u) => serializer.serialize_u64(*u),
            Json::Float(f) => serializer.serialize_f64(*f),
            Json::String(s) => serializer.serialize_str(s),
            Json::Array(items) => serializer.collect_seq(items),
            Json::Object(members) => serializer.collect_map(members.iter().map(|(k, v)| (k, v))),
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
        Ok(Json::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
        Ok(Json::Int(v))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
        Ok(i64::try_from(v).map_or(Json::Uint(v), Json::Int))
    }

    fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
        Ok(Json::Float(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Json, E> {
        Ok(Json::String(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Json, E> {
        Ok(Json::String(v))
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_none<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Json, D::Error> {
        Json::deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        // The size hint comes from the input, so it is capped: a hostile
        // document must not be able to reserve memory it never fills.
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(1024));
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Json::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut members = Vec::with_capacity(map.size_hint().unwrap_or(0).min(1024));
        while let Some(member) = map.next_entry::<String, Json>()? {
            members.push(member);
        }
        Ok(Json::Object(members))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn a_document_survives_a_round_trip_byte_for_byte() {
        // Key order (b before a), a repeated key, both integer extremes, a
        // float, an escaped non-ASCII character and null: everything a tool
        // schema or a tool call can contain.
        let text = r#"{"b":1,"a":[-9223372036854775808,18446744073709551615,2.5,"ä\n",null,true],"b":{"x":[]}}"#;
        let json = parse(text);
        assert_eq!(
            serde_json::to_string(&json).unwrap(),
            r#"{"b":1,"a":[-9223372036854775808,18446744073709551615,2.5,"ä\n",null,true],"b":{"x":[]}}"#
        );
    }

    #[test]
    fn integers_have_one_representation_and_floats_stay_floats() {
        assert_eq!(parse("5"), Json::Int(5));
        assert_eq!(parse("9223372036854775807"), Json::Int(i64::MAX));
        assert_eq!(parse("9223372036854775808"), Json::Uint(1 << 63));
        assert_eq!(parse("5.0"), Json::Float(5.0));
        assert_eq!(serde_json::to_string(&parse("5.0")).unwrap(), "5.0");
    }

    #[test]
    fn objects_keep_document_order_and_get_returns_the_last_duplicate() {
        let json = parse(r#"{"z":1,"a":2,"z":3}"#);
        let keys: Vec<_> = json.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["z", "a", "z"]);
        assert_eq!(json.get("z"), Some(&Json::Int(3)));
        assert_eq!(json.get("missing"), None);
        assert_eq!(Json::Int(1).get("z"), None);
    }

    #[test]
    fn accessors_only_match_their_own_kind() {
        assert_eq!(parse(r#""s""#).as_str(), Some("s"));
        assert_eq!(parse("1").as_str(), None);
        assert_eq!(parse("[1]").as_array().map(<[Json]>::len), Some(1));
        assert!(parse("null").is_null());
        assert!(!parse("0").is_null());
        assert_eq!(parse("{}").kind(), "an object");
    }
}
