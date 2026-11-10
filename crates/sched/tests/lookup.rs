//! A randomised check of the lookup drafter against a search of all text.
//!
//! Several sequences write tokens from a small alphabet in a random
//! interleaving, so that everything repeats and most contexts went on in
//! more than one way. The reference is the definition itself, computed by
//! comparing the end of a sequence with every earlier place in every text:
//! the longest suffix that occurred before and went on, how many of its
//! occurrences went on, and what followed the one that went on last.
//!
//! The first test holds the automaton to that exactly (one automaton, walks
//! without a cap). The second runs with little memory, a capped walk and
//! rollbacks, where exact agreement is not promised, and checks what is: a
//! draft is always text that really followed an earlier occurrence of the
//! reported match, memory stays within its bound, and text that was written
//! within one automaton's capacity is still found.

use ares_sched::{LookupConfig, LookupCorpus, LookupDrafter, LookupMatch};

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// One sequence: its tokens, each with the time it was written.
#[derive(Default)]
struct Text {
    tokens: Vec<u32>,
    times: Vec<u64>,
}

/// The number of tokens the ends of `a` and `b` have in common.
fn common_suffix(a: &[u32], b: &[u32]) -> usize {
    a.iter().rev().zip(b.iter().rev()).take_while(|(x, y)| x == y).count()
}

/// What a search finds for the end of `texts[own]`: the match, and the draft
/// of up to `max` tokens.
fn search(
    texts: &[Text],
    own: usize,
    min_match: usize,
    max: usize,
) -> Option<(LookupMatch, Vec<u32>)> {
    let end = &texts[own].tokens;
    // Every place that went on, with the length of its match.
    let mut places: Vec<(usize, usize, usize)> = Vec::new();
    for (j, text) in texts.iter().enumerate() {
        for e in 0..text.tokens.len().saturating_sub(1) {
            let shared = common_suffix(end, &text.tokens[..=e]);
            if shared >= min_match {
                places.push((shared, j, e));
            }
        }
    }
    let len = places.iter().map(|p| p.0).max()?;
    places.retain(|p| p.0 == len);
    // The occurrence whose continuation was written last.
    let (_, j, e) = *places.iter().max_by_key(|(_, j, e)| texts[*j].times[e + 1])?;
    let first = texts[j].tokens[e + 1];
    let agreeing = places
        .iter()
        .filter(|(_, j, e)| texts[*j].tokens[e + 1] == first && e + 2 < texts[*j].tokens.len())
        .count();
    let draft = texts[j].tokens[e + 1..].iter().take(max).copied().collect();
    let found = LookupMatch {
        len: len as u32,
        occurrences: places.len() as u32,
        agreeing: agreeing.min(places.len()) as u32,
    };
    Some((found, draft))
}

#[test]
fn the_automaton_finds_what_a_search_of_all_text_finds() {
    let mut compared = 0u64;
    let mut longest = 0u32;
    for (seed, alphabet, min_match) in
        [(1u64, 2u64, 3u32), (2, 3, 2), (3, 4, 3), (4, 6, 1), (5, 3, 5)]
    {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let config = LookupConfig { max_bytes: 64 << 20, min_match, max_hops: u32::MAX };
        let mut corpus = LookupCorpus::new(config);
        let mut texts: Vec<Text> = Vec::new();
        let mut drafters: Vec<LookupDrafter> = Vec::new();
        let mut clock = 0u64;
        for _ in 0..2500 {
            // A new sequence now and then; otherwise one of them writes.
            if drafters.is_empty() || (drafters.len() < 6 && rng.below(60) == 0) {
                drafters.push(LookupDrafter::new());
                texts.push(Text::default());
            }
            let i = rng.below(drafters.len() as u64) as usize;
            // Mostly it copies what the search proposes, so that long
            // matches build up; sometimes it writes something else.
            let proposal = search(&texts, i, min_match as usize, 1).map(|(_, draft)| draft[0]);
            let token = match proposal {
                Some(token) if rng.below(4) != 0 => token,
                _ => rng.below(alphabet) as u32,
            };
            clock += 1;
            drafters[i].push(&mut corpus, token);
            texts[i].tokens.push(token);
            texts[i].times.push(clock);

            // The end of any sequence, not only the one that just wrote.
            let k = rng.below(drafters.len() as u64) as usize;
            let max = 1 + rng.below(12) as usize;
            let expected = search(&texts, k, min_match as usize, max);
            let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
            let len = drafters[k].draft(&corpus, max, &mut tokens, &mut confidence);
            let found = drafters[k].matched(&corpus);
            match expected {
                None => assert!(found.is_none() && tokens.is_empty(), "seed {seed}: {tokens:?}"),
                Some((expected, draft)) => {
                    assert_eq!(found, Some(expected), "seed {seed}, sequence {k}");
                    assert_eq!(tokens, draft, "seed {seed}, sequence {k}");
                    assert_eq!(len as u32, expected.len);
                    longest = longest.max(expected.len);
                    compared += 1;
                }
            }
        }
        assert_eq!(corpus.tokens(), texts.iter().map(|t| t.tokens.len()).sum::<usize>());
    }
    assert!(compared > 8000, "only {compared} matches were compared");
    assert!(longest > 30, "the longest match was {longest} tokens");
}

#[test]
fn drafts_are_real_text_with_little_memory_a_capped_walk_and_rollbacks() {
    let config = LookupConfig { max_bytes: 192 << 10, min_match: 3, max_hops: 6 };
    let mut corpus = LookupCorpus::new(config);
    let per_generation = corpus.tokens_per_generation();
    assert!(corpus.bytes() <= config.max_bytes);
    let mut rng = Rng(0xfeed_f00d);
    // Everything every sequence ever wrote, as pieces of text: a rollback
    // leaves the piece as it was (with the removed tokens) and continues in
    // a copy of what was kept. A draft may come from any piece.
    let mut written: Vec<Vec<u32>> = Vec::new();
    let mut texts: Vec<Vec<u32>> = Vec::new();
    let mut piece: Vec<usize> = Vec::new();
    let mut drafters: Vec<LookupDrafter> = Vec::new();
    let (mut drafts, mut found_recent, mut asked_recent) = (0u64, 0u64, 0u64);
    let mut pushed = 0usize;
    while pushed < 12 * per_generation {
        if drafters.len() < 5 && (drafters.is_empty() || rng.below(200) == 0) {
            drafters.push(LookupDrafter::new());
            texts.push(Vec::new());
            piece.push(written.len());
            written.push(Vec::new());
        }
        let i = rng.below(drafters.len() as u64) as usize;
        match rng.below(100) {
            // A rollback of a few tokens, or of many.
            0..=2 if !texts[i].is_empty() => {
                let back = if rng.below(4) == 0 { rng.below(80) } else { rng.below(6) } as usize;
                let len = texts[i].len().saturating_sub(back);
                drafters[i].truncate(&corpus, len);
                texts[i].truncate(len);
                piece[i] = written.len();
                written.push(texts[i].clone());
            }
            // A sequence ends; its text stays in the corpus.
            3 if drafters.len() > 1 => {
                drafters.swap_remove(i);
                texts.swap_remove(i);
                piece.swap_remove(i);
            }
            _ => {
                // A passage that was written a moment ago is written again.
                let token = if rng.below(3) == 0 && texts[i].len() > 40 {
                    let from = texts[i].len() - 30;
                    texts[i][from + rng.below(10) as usize]
                } else {
                    rng.below(8) as u32
                };
                drafters[i].push(&mut corpus, token);
                texts[i].push(token);
                written[piece[i]].push(token);
                pushed += 1;
            }
        }
        assert!(corpus.tokens() <= 2 * per_generation);
        for (drafter, text) in drafters.iter().zip(&texts) {
            assert_eq!(drafter.len(), text.len());
        }

        let k = rng.below(drafters.len() as u64) as usize;
        let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
        let len = drafters[k].draft(&corpus, 8, &mut tokens, &mut confidence);
        if tokens.is_empty() {
            continue;
        }
        drafts += 1;
        // The match and the draft together occurred somewhere, in that order.
        assert!(len >= 3 && len <= texts[k].len());
        let mut needle = texts[k][texts[k].len() - len..].to_vec();
        needle.extend_from_slice(&tokens);
        assert!(
            written.iter().any(|text| text.windows(needle.len()).any(|w| w == needle)),
            "a draft that never followed its match: {needle:?}"
        );
        drafters[k].observe(&mut corpus, tokens.len(), rng.below(tokens.len() as u64 + 1) as usize);
    }
    assert!(drafts > 1000, "{drafts} drafts");

    // Text written within one automaton's capacity is found: a sequence
    // writes a passage of new tokens, others write as much as fits in the
    // rest of an automaton, and the passage is found again.
    for round in 0..20u32 {
        let passage: Vec<u32> = (0..24).map(|i| 1_000_000 + round * 100 + i).collect();
        let mut writer = LookupDrafter::new();
        writer.extend(&mut corpus, &passage);
        let mut other = LookupDrafter::new();
        for _ in 0..per_generation - 64 {
            other.push(&mut corpus, rng.below(40) as u32);
        }
        let mut reader = LookupDrafter::new();
        reader.extend(&mut corpus, &passage[..6]);
        let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
        asked_recent += 1;
        if reader.draft(&corpus, 4, &mut tokens, &mut confidence) > 0 {
            assert_eq!(tokens, passage[6..10]);
            found_recent += 1;
        }
    }
    assert_eq!(found_recent, asked_recent, "text within one capacity was forgotten");
}

#[test]
fn text_with_one_place_to_continue_from_is_drafted_from_the_third_token_on() {
    // Text without ambiguity: a long run of distinct tokens, repeated. The
    // drafter must propose the continuation at every point of the second
    // pass, from the third token on, which is the shortest match here.
    let passage: Vec<u32> = (0..200).map(|i| (i * 7919) % 1000 + 10).collect();
    let config = LookupConfig { min_match: 3, ..LookupConfig::default() };
    let mut corpus = LookupCorpus::new(config);
    let mut lookup = LookupDrafter::new();
    lookup.extend(&mut corpus, &passage);
    let (mut tokens, mut confidence) = (Vec::new(), Vec::new());
    for (i, token) in passage.iter().enumerate().take(150) {
        lookup.push(&mut corpus, *token);
        lookup.draft(&corpus, 8, &mut tokens, &mut confidence);
        if i >= 2 {
            assert_eq!(tokens, passage[i + 1..i + 9], "after {} tokens of the second pass", i + 1);
        } else {
            assert!(tokens.is_empty());
        }
    }
}
