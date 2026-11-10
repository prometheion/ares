//! Byte-pair encoding of one word.
//!
//! A word starts as its bytes, each already a token of its own. Then the pair of
//! neighbours with the lowest merge rank is joined into the token the merge
//! names, and again, until no neighbours have a merge. When several pairs have
//! the lowest rank, the leftmost goes first. This is what llama.cpp computes
//! with a priority queue over the pairs, and the result depends on nothing else.
//!
//! The merges are held as ids: `(left, right) -> (rank, merged)`. That is
//! exactly equivalent to llama.cpp's merging of strings, because loading checks
//! that every merge whose two sides are tokens produces a token (the one case
//! where the string version would go on with a symbol that is no token, and
//! then drop bytes).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A merge: its position in the model's list (lower goes first) and the token
/// it produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Merge {
    pub(crate) rank: u32,
    pub(crate) token: u32,
}

/// The merges, in a hash table keyed by the two ids of a pair.
///
/// Open addressing with linear probing. A key that is all ones marks an empty
/// slot, which no pair of valid ids can be.
pub(crate) struct Merges {
    slots: Box<[Slot]>,
    shift: u32,
}

#[derive(Clone, Copy)]
struct Slot {
    key: u64,
    rank: u32,
    token: u32,
}

const EMPTY: u64 = u64::MAX;

#[inline(always)]
fn key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

impl Merges {
    /// A table with room for `count` merges.
    pub(crate) fn with_capacity(count: usize) -> Self {
        let slots = (count * 2).next_power_of_two().max(16);
        Self {
            slots: vec![Slot { key: EMPTY, rank: 0, token: 0 }; slots].into(),
            shift: 64 - slots.trailing_zeros(),
        }
    }

    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// Adds a merge unless the pair already has one: the first merge of a pair
    /// is the one that counts. Returns whether the merge was added.
    pub(crate) fn insert(&mut self, left: u32, right: u32, merge: Merge) -> bool {
        let key = key(left, right);
        let mask = self.slots.len() - 1;
        let mut i = self.home(key);
        loop {
            let slot = &mut self.slots[i];
            if slot.key == EMPTY {
                *slot = Slot { key, rank: merge.rank, token: merge.token };
                return true;
            }
            if slot.key == key {
                return false;
            }
            i = (i + 1) & mask;
        }
    }

    /// The merge of the pair `(left, right)`, if there is one.
    #[inline(always)]
    pub(crate) fn get(&self, left: u32, right: u32) -> Option<Merge> {
        let key = key(left, right);
        let mask = self.slots.len() - 1;
        let mut i = self.home(key);
        loop {
            let slot = &self.slots[i];
            if slot.key == key {
                return Some(Merge { rank: slot.rank, token: slot.token });
            }
            if slot.key == EMPTY {
                return None;
            }
            i = (i + 1) & mask;
        }
    }
}

/// Words up to this many bytes are merged by repeated scans of a small array,
/// longer ones with a heap of pairs. Both give the same tokens; the scan is
/// quicker for the short words that make up almost all text, the heap does not
/// slow down quadratically on a very long word.
const SCAN_LIMIT: usize = 48;
/// The arrays of the scan come in two sizes, so that the common short word does
/// not pay for clearing the long one.
const SHORT: usize = 12;

/// Appends the tokens of `word` to `out`. `byte_token[b]` is the token of the
/// single byte `b`.
pub(crate) fn encode_word(
    merges: &Merges,
    byte_token: &[u32; 256],
    word: &[u8],
    out: &mut Vec<u32>,
) {
    match word.len() {
        0 => {}
        1 => out.push(byte_token[usize::from(word[0])]),
        n if n <= SHORT => encode_by_scan::<SHORT>(merges, byte_token, word, out),
        n if n <= SCAN_LIMIT => encode_by_scan::<SCAN_LIMIT>(merges, byte_token, word, out),
        _ => encode_by_heap(merges, byte_token, word, out),
    }
}

fn encode_by_scan<const N: usize>(
    merges: &Merges,
    byte_token: &[u32; 256],
    word: &[u8],
    out: &mut Vec<u32>,
) {
    let mut ids = [0u32; N];
    // For the pair (ids[i], ids[i + 1]): its rank, or NONE, and its merged token.
    let mut ranks = [NONE; N];
    let mut merged = [0u32; N];
    let mut len = word.len();
    for (id, byte) in ids.iter_mut().zip(word) {
        *id = byte_token[usize::from(*byte)];
    }
    let look = |ids: &[u32; N], i: usize| match merges.get(ids[i], ids[i + 1]) {
        Some(m) => (m.rank, m.token),
        None => (NONE, 0),
    };
    for i in 0..len - 1 {
        (ranks[i], merged[i]) = look(&ids, i);
    }
    loop {
        // The leftmost pair of the lowest rank.
        let mut best = NONE;
        let mut at = 0;
        for (i, &rank) in ranks[..len - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                at = i;
            }
        }
        if best == NONE {
            break;
        }
        ids[at] = merged[at];
        ids.copy_within(at + 2..len, at + 1);
        // The pairs to the right of the merged one move left with their tokens.
        if at + 2 < len - 1 {
            ranks.copy_within(at + 2..len - 1, at + 1);
            merged.copy_within(at + 2..len - 1, at + 1);
        }
        len -= 1;
        if len == 1 {
            break;
        }
        // Only the pairs next to the new token changed.
        if at > 0 {
            (ranks[at - 1], merged[at - 1]) = look(&ids, at - 1);
        }
        if at + 1 < len {
            (ranks[at], merged[at]) = look(&ids, at);
        }
    }
    out.extend_from_slice(&ids[..len]);
}

/// No merge. Ranks are below this: a model has far fewer merges than `u32::MAX`.
const NONE: u32 = u32::MAX;

fn encode_by_heap(merges: &Merges, byte_token: &[u32; 256], word: &[u8], out: &mut Vec<u32>) {
    let n = word.len();
    let mut ids: Vec<u32> = word.iter().map(|b| byte_token[usize::from(*b)]).collect();
    // Neighbours as a doubly linked list; a token that was merged into its left
    // neighbour is dead.
    let mut next: Vec<u32> = (1..=n as u32).collect();
    let mut prev: Vec<u32> = (0..n as u32).map(|i| i.wrapping_sub(1)).collect();
    let mut alive = vec![true; n];
    // (rank, left position): the lowest rank first, the leftmost among equals.
    let mut queue: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::with_capacity(n);
    for i in 0..n - 1 {
        if let Some(m) = merges.get(ids[i], ids[i + 1]) {
            queue.push(Reverse((m.rank, i as u32)));
        }
    }
    while let Some(Reverse((rank, left))) = queue.pop() {
        let left = left as usize;
        let right = next[left] as usize;
        // An entry is stale when its left token was merged away or changed, or
        // when its right neighbour is no longer the one it was made for. The
        // rank names the pair, so comparing it is enough.
        if !alive[left] || right >= n {
            continue;
        }
        match merges.get(ids[left], ids[right]) {
            Some(m) if m.rank == rank => ids[left] = m.token,
            _ => continue,
        }
        alive[right] = false;
        next[left] = next[right];
        if (next[right] as usize) < n {
            prev[next[right] as usize] = left as u32;
        }
        let before = prev[left];
        if before != u32::MAX
            && let Some(m) = merges.get(ids[before as usize], ids[left])
        {
            queue.push(Reverse((m.rank, before)));
        }
        let after = next[left] as usize;
        if after < n
            && let Some(m) = merges.get(ids[left], ids[after])
        {
            queue.push(Reverse((m.rank, left as u32)));
        }
    }
    let mut i = 0;
    while i < n {
        out.push(ids[i]);
        i = next[i] as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte `b` is token `b`; merges are given as (left, right, token) in rank order.
    fn table(list: &[(u32, u32, u32)]) -> Merges {
        let mut merges = Merges::with_capacity(list.len());
        for (rank, &(left, right, token)) in list.iter().enumerate() {
            merges.insert(left, right, Merge { rank: rank as u32, token });
        }
        merges
    }

    fn identity() -> [u32; 256] {
        std::array::from_fn(|i| i as u32)
    }

    const A: u32 = b'a' as u32;
    const B: u32 = b'b' as u32;
    const C: u32 = b'c' as u32;

    fn run(merges: &Merges, word: &[u8], by_heap: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if by_heap {
            encode_by_heap(merges, &identity(), word, &mut out);
        } else {
            encode_by_scan::<SCAN_LIMIT>(merges, &identity(), word, &mut out);
        }
        out
    }

    #[test]
    fn joins_the_lowest_rank_pair_first() {
        // ab (300), bc (301), abc = ab + c (302) comes after bc: for "abc" the
        // pair (b, c) has rank 0 and (a, b) rank 1.
        let merges = table(&[(B, C, 301), (A, B, 300), (300, C, 302), (A, 301, 303)]);
        for by_heap in [false, true] {
            assert_eq!(run(&merges, b"abc", by_heap), [303], "heap: {by_heap}");
            assert_eq!(run(&merges, b"ab", by_heap), [300]);
            assert_eq!(run(&merges, b"cab", by_heap), [C, 300]);
            assert_eq!(run(&merges, b"xyz", by_heap), [120, 121, 122]);
        }
    }

    #[test]
    fn takes_the_leftmost_of_equal_pairs_and_does_not_overlap_them() {
        // aa = 300, aaa has no merge: "aaa" is aa + a, never a + aa.
        let merges = table(&[(A, A, 300)]);
        for by_heap in [false, true] {
            assert_eq!(run(&merges, b"aaa", by_heap), [300, A]);
            assert_eq!(run(&merges, b"aaaa", by_heap), [300, 300]);
            assert_eq!(run(&merges, b"baaab", by_heap), [B, 300, A, B]);
        }
    }

    /// The definition, written for clarity: find the leftmost pair of the lowest
    /// rank, join it, repeat.
    fn naive(merges: &Merges, word: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = word.iter().map(|b| u32::from(*b)).collect();
        loop {
            let best = (0..ids.len().saturating_sub(1))
                .filter_map(|i| merges.get(ids[i], ids[i + 1]).map(|m| (m.rank, i, m.token)))
                .min();
            let Some((_, at, token)) = best else { return ids };
            ids[at] = token;
            ids.remove(at + 1);
        }
    }

    #[test]
    fn the_scan_and_the_heap_give_what_the_definition_gives() {
        // A small alphabet makes merges likely; the merge list is a fixed
        // pseudo-random one over ids that grow as merges chain.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut list = Vec::new();
        for i in 0..400u32 {
            let side = |next: &mut dyn FnMut(u64) -> u64| {
                if i < 20 || next(2) == 0 {
                    97 + next(4) as u32
                } else {
                    300 + next(u64::from(i) - 1) as u32
                }
            };
            let (l, r) = (side(&mut next), side(&mut next));
            list.push((l, r, 300 + i));
        }
        let merges = table(&list);
        for _ in 0..3000 {
            let len = 1 + next(90) as usize;
            let word: Vec<u8> = (0..len).map(|_| b'a' + next(4) as u8).collect();
            let expected = naive(&merges, &word);
            assert_eq!(run(&merges, &word, true), expected, "heap: {word:?}");
            if len <= SCAN_LIMIT {
                assert_eq!(run(&merges, &word, false), expected, "scan: {word:?}");
            }
            let mut out = Vec::new();
            encode_word(&merges, &identity(), &word, &mut out);
            assert_eq!(out, expected, "encode_word: {word:?}");
        }
    }

    #[test]
    fn keeps_the_first_merge_of_a_pair() {
        let mut merges = Merges::with_capacity(4);
        assert!(merges.insert(A, B, Merge { rank: 3, token: 300 }));
        assert!(!merges.insert(A, B, Merge { rank: 4, token: 301 }));
        assert_eq!(merges.get(A, B), Some(Merge { rank: 3, token: 300 }));
        assert_eq!(merges.get(B, A), None);
    }

    #[test]
    fn handles_words_longer_than_the_scan_limit() {
        let merges = table(&[(A, A, 300), (300, 300, 301)]);
        let word = vec![b'a'; 1001];
        let mut out = Vec::new();
        encode_word(&merges, &identity(), &word, &mut out);
        assert_eq!(out.len(), 251, "250 tokens of 4 and one of 1");
        assert_eq!(&out[..250], &[301; 250][..]);
        assert_eq!(out[250], A);
    }
}
