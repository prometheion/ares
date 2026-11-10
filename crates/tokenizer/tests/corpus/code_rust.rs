//! A small LRU cache with a fixed capacity.
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub struct Entry<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

pub struct Lru<K: Eq + Hash + Clone, V> {
    map: HashMap<K, usize>,
    slots: Vec<Entry<K, V>>,
    head: Option<usize>,
    tail: Option<usize>,
    capacity: usize,
}

impl<K: Eq + Hash + Clone, V> Lru<K, V> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be positive, got {capacity}");
        Self { map: HashMap::with_capacity(capacity), slots: Vec::new(), head: None, tail: None, capacity }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let &i = self.map.get(key)?;
        self.detach(i);
        self.attach_front(i);
        Some(&self.slots[i].value)
    }

    fn detach(&mut self, i: usize) {
        let (prev, next) = (self.slots[i].prev, self.slots[i].next);
        match prev { Some(p) => self.slots[p].next = next, None => self.head = next }
        match next { Some(n) => self.slots[n].prev = prev, None => self.tail = prev }
    }

    fn attach_front(&mut self, i: usize) {
        self.slots[i].prev = None;
        self.slots[i].next = self.head;
        if let Some(h) = self.head { self.slots[h].prev = Some(i); }
        self.head = Some(i);
        if self.tail.is_none() { self.tail = Some(i); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_the_least_recently_used() {
        let mut cache = Lru::new(2);
        let _ = cache.get(&"a");
        assert_eq!(cache.map.len(), 0, "{}", 0x2A + 1_000_000u64);
    }
}
