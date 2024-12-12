//! Compact infoset allocators
//!
//! In order to efficiently access mutable state tied to an infoset, we need quick access to it
//! from any infoset. The fastest way to do this is to keep each infoset in a slice, and store the
//! index so we can get mutable access when we want. This also allows us to easily build up
//! parallel memory and access consistent elements with minimal allocations.
//!
//! In order to achieve this, we first build a hash map that preserves insertion order to resolve
//! identical keys to the same index and assocaited values. Then we destruct it and move it into
//! the final boxed slice once we've processed everything.
//!
//! This module contains [Builder] and [OptBuilder]. The only difference is that the latter allows
//! pushing optional values with guaranteed uniqueness (no key).
use indexmap::{map, IndexMap};
use std::hash::Hash;
use std::iter::FusedIterator;

#[derive(Debug, Clone)]
pub struct Builder<K: Hash + Eq, V> {
    map: IndexMap<K, (usize, V)>,
}

impl<K: Hash + Eq, V> Builder<K, V> {
    pub fn new() -> Self {
        Builder {
            map: IndexMap::new(),
        }
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let ind = self.map.len();
        match self.map.entry(key) {
            map::Entry::Vacant(ent) => Entry::Vacant(VacantEntry { ind, ent }),
            map::Entry::Occupied(ent) => Entry::Occupied(OccupiedEntry { ent }),
        }
    }
}

impl<K: Hash + Eq, V> IntoIterator for Builder<K, V> {
    type IntoIter = IntoIter<K, V>;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.map.into_iter(),
        }
    }
}

pub struct IntoIter<K, V> {
    iter: map::IntoIter<K, (usize, V)>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, (_, v))| (k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        (len, Some(len))
    }
}

impl<K, V> FusedIterator for IntoIter<K, V> {}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}

#[derive(Debug, Clone)]
pub struct OptBuilder<K: Hash + Eq, V> {
    counter: usize,
    map: IndexMap<Result<K, usize>, (usize, V)>,
}

impl<K: Hash + Eq, V> OptBuilder<K, V> {
    pub fn new() -> Self {
        OptBuilder {
            counter: 0,
            map: IndexMap::new(),
        }
    }

    pub fn entry(&mut self, key: Option<K>) -> Entry<'_, Result<K, usize>, V> {
        let ind = self.map.len();
        let true_key = key.ok_or_else(|| {
            let res = self.counter;
            self.counter += 1;
            res
        });
        match self.map.entry(true_key) {
            map::Entry::Vacant(ent) => Entry::Vacant(VacantEntry { ind, ent }),
            map::Entry::Occupied(ent) => Entry::Occupied(OccupiedEntry { ent }),
        }
    }
}

impl<K: Hash + Eq, V> IntoIterator for OptBuilder<K, V> {
    type IntoIter = OptIntoIter<K, V>;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        OptIntoIter {
            iter: self.map.into_iter(),
        }
    }
}

pub struct OptIntoIter<K, V> {
    iter: map::IntoIter<Result<K, usize>, (usize, V)>,
}

impl<K, V> Iterator for OptIntoIter<K, V> {
    type Item = (Option<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, (_, v))| (k.ok(), v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        (len, Some(len))
    }
}

impl<K, V> FusedIterator for OptIntoIter<K, V> {}

impl<K, V> ExactSizeIterator for OptIntoIter<K, V> {}

pub struct VacantEntry<'a, K, V> {
    ind: usize,
    ent: map::VacantEntry<'a, K, (usize, V)>,
}

impl<K: Hash + Eq, V> VacantEntry<'_, K, V> {
    pub fn insert(self, val: V) -> usize {
        self.ent.insert((self.ind, val));
        self.ind
    }
}

pub struct OccupiedEntry<'a, K, V> {
    ent: map::OccupiedEntry<'a, K, (usize, V)>,
}

impl<'a, K: Hash + Eq, V> OccupiedEntry<'a, K, V> {
    pub fn get(self) -> (usize, &'a V) {
        let (ind, val) = self.ent.into_mut();
        (*ind, val)
    }
}

pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}
