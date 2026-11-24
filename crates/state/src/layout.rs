//! What a page costs.

/// The size of one page on each device, built up from the parts of a model's
/// state that live in pages.
///
/// A model describes its state once (which layers keep K and V rows, in which
/// format, on which device; which keep keys per block), and the page capacity
/// of the pool follows from the memory budgets. Nothing else decides how much
/// context fits.
#[derive(Clone, Debug)]
pub struct PageLayout {
    cells_per_page: u32,
    bytes_per_page: Vec<u64>,
    parts: Vec<(usize, String, u64)>,
}

impl PageLayout {
    /// A layout for `n_devices` devices with pages of `cells_per_page` cells.
    ///
    /// A model with a block-wise attention indexer chooses the page size as a
    /// multiple of the indexer's block, so a selected block never straddles
    /// pages.
    pub fn new(n_devices: usize, cells_per_page: u32) -> Self {
        assert!(cells_per_page > 0, "a page holds at least one cell");
        Self { cells_per_page, bytes_per_page: vec![0; n_devices], parts: Vec::new() }
    }

    pub fn cells_per_page(&self) -> u32 {
        self.cells_per_page
    }

    /// Adds a part that stores `bytes_per_cell` for every cell on `device`,
    /// such as the K and V rows of the attention heads that device holds.
    pub fn per_cell(mut self, device: usize, name: &str, bytes_per_cell: u64) -> Self {
        let bytes = bytes_per_cell * u64::from(self.cells_per_page);
        self.bytes_per_page[device] += bytes;
        self.parts.push((device, name.to_owned(), bytes));
        self
    }

    /// Adds a part that stores `bytes_per_page` once per page on `device`,
    /// such as one pooled indexer key per block of cells.
    pub fn per_page(mut self, device: usize, name: &str, bytes_per_page: u64) -> Self {
        self.bytes_per_page[device] += bytes_per_page;
        self.parts.push((device, name.to_owned(), bytes_per_page));
        self
    }

    /// The bytes one page takes on `device`.
    pub fn bytes_per_page(&self, device: usize) -> u64 {
        self.bytes_per_page[device]
    }

    /// The parts as `(device, name, bytes per page)`, in the order added.
    pub fn parts(&self) -> impl Iterator<Item = (usize, &str, u64)> {
        self.parts.iter().map(|(device, name, bytes)| (*device, name.as_str(), *bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state of Qwen3.8-Flash-Next (`qwen4exp`) as Strata stores it: 12
    /// attention layers, 2 KV heads of 256 values, K and V as int8 with an
    /// f16 scale per 64 values (1 056 bytes per cell and layer), one f32 key
    /// of 128 values per block of 4 cells.
    #[test]
    fn sizes_the_context_of_a_qwen4exp_shaped_model() {
        let layers = 12u64;
        let kv_per_cell = 2 * 2 * (256 + 256 / 64 * 2);
        assert_eq!(kv_per_cell, 1056);
        let layout = PageLayout::new(1, 4)
            .per_cell(0, "K and V, 12 attention layers", layers * kv_per_cell)
            .per_page(0, "pooled indexer keys", layers * 128 * 4);
        assert_eq!(layout.bytes_per_page(0), 4 * 12_672 + 6_144);

        // The model's full context of 262 144 tokens.
        let full_context = 262_144 / 4 * layout.bytes_per_page(0);
        assert_eq!(full_context, 3_724_541_952, "3.47 GiB");
    }

    #[test]
    fn each_device_has_the_bytes_of_its_parts() {
        // An attention split of 1:3 KV heads over two devices.
        let layout =
            PageLayout::new(2, 64).per_cell(0, "KV head 0", 100).per_cell(1, "KV heads 1-3", 300);
        assert_eq!((layout.bytes_per_page(0), layout.bytes_per_page(1)), (6_400, 19_200));
        let parts: Vec<_> = layout.parts().collect();
        assert_eq!(parts, [(0, "KV head 0", 6_400), (1, "KV heads 1-3", 19_200)]);
    }
}
