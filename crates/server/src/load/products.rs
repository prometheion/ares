//! What the weight kernels of a device were measured to be when the model was
//! loaded, for the log: whether its tensor cores add f16 products into f32 at
//! full rate, which path of the tile kernels each weight type of the file's
//! layout takes, and which kernels of the device layout serve the rows above a
//! decode step's. Every card is measured, capped or not, and the faster path
//! runs; there is no option for another (issue #54).

use ares_cuda::{LayoutTable, PrefillChoice, PrefillPath};

use super::join;

/// The log's line of one device, without its number.
pub(crate) fn products_line(
    full_rate: bool,
    paths: &[PrefillChoice],
    tables: &[LayoutTable],
) -> String {
    let rate = if full_rate {
        "the tensor cores add f16 products into f32 at full rate"
    } else {
        "the tensor cores add f16 products into f32 at half the rate of f16 sums (the cap of \
         GeForce cards, which gpu-unlock lifts)"
    };
    let mut parts = vec![rate.to_owned()];
    if !paths.is_empty() {
        parts.push(format!(
            "tile kernels of the file's layout per weight type (a probe product in ms: int8 / \
             f16 / f16 with f16 sums): {}",
            join(paths.iter().map(path_text), ", ")
        ));
    }
    if !tables.is_empty() {
        parts.push(format!(
            "kernels of the device layout above a decode step's rows: {}",
            join(tables.iter().map(table_text), ", ")
        ));
    }
    parts.join("; ")
}

fn path_name(path: PrefillPath) -> &'static str {
    match path {
        PrefillPath::Int8 => "int8",
        PrefillPath::F16 => "f16",
        PrefillPath::F16Flush => "f16 with f16 sums",
    }
}

fn path_text(choice: &PrefillChoice) -> String {
    let name = format!("{} {}", choice.ty.name(), path_name(choice.path));
    if !choice.measured {
        return format!("{name} (not measured: the kept rule)");
    }
    let ms = |ms: f32| if ms.is_finite() { format!("{ms:.3}") } else { "-".to_owned() };
    let [int8, f16, flushed] = choice.probe_ms;
    format!("{name} ({} / {} / {})", ms(int8), ms(f16), ms(flushed))
}

fn table_text(table: &LayoutTable) -> String {
    let rows = table.rows.iter().map(|bucket| {
        let up_to = match bucket.up_to_rows {
            u32::MAX => "above".to_owned(),
            rows => format!("to {rows} rows"),
        };
        let (fast, exact) = (&bucket.fast, &bucket.exact);
        let sums = if fast.f32_accumulation { "f32" } else { "f16" };
        let mut text = format!("{} {up_to} ({sums} sums", fast.name);
        if table.measured {
            text.push_str(&format!(", {:.3} ms", fast.probe_ms));
            if !fast.f32_accumulation {
                text.push_str(&format!(", with f32 sums {:.3}", exact.probe_ms));
            }
        }
        text.push(')');
        text
    });
    let measured = if table.measured { "" } else { " (not measured: the kept table)" };
    format!("{} {}{measured}", table.ty.name(), join(rows, ", "))
}

#[cfg(test)]
mod tests {
    use ares_core::GgmlType;

    use super::*;

    #[test]
    fn the_line_names_the_rate_and_what_each_type_was_measured_to_take() {
        let measured = PrefillChoice {
            ty: GgmlType::Q4_K,
            measured: true,
            path: PrefillPath::F16Flush,
            probe_ms: [0.92, 1.8, 0.9],
        };
        let floats = PrefillChoice {
            ty: GgmlType::BF16,
            measured: true,
            path: PrefillPath::F16,
            probe_ms: [f32::INFINITY, 1.1, 1.12],
        };
        let kept = PrefillChoice {
            ty: GgmlType::Q6_K,
            measured: false,
            path: PrefillPath::Int8,
            probe_ms: [0.0; 3],
        };
        let line = products_line(false, &[measured, floats, kept], &[]);
        assert_eq!(
            line,
            "the tensor cores add f16 products into f32 at half the rate of f16 sums (the cap of \
             GeForce cards, which gpu-unlock lifts); tile kernels of the file's layout per weight \
             type (a probe product in ms: int8 / f16 / f16 with f16 sums): q4_K f16 with f16 \
             sums (0.920 / 1.800 / 0.900), bf16 f16 (- / 1.100 / 1.120), q6_K int8 (not \
             measured: the kept rule)"
        );
        assert_eq!(
            products_line(true, &[], &[]),
            "the tensor cores add f16 products into f32 at full rate"
        );
    }
}
