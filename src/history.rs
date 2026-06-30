//! Cheap, hashable representations of an append-only game history, for use as a [`Game::Infoset`] key.
//!
//! Both fold each appended action into a rolling hash so the resulting key hashes in O(1) instead of
//! re-walking the whole history every time the solver looks an infoset up.
//!
//! [`History`] keeps the full history behind shared structure: cloning and appending are O(1), the
//! cached digest makes hashing O(1), and equality is exact (a digest mismatch settles it instantly,
//! a match falls back to a structural compare so collisions stay correct). Reach for it when the
//! infoset must be compared exactly -- which is almost always.
//!
//! [`Digest`] keeps *only* the rolling hash -- `N` independent 64-bit lanes, `Digest<1>` by default
//! and wider for smaller collision odds. It is [`Copy`], allocates nothing, and is the cheapest
//! possible key, but it is lossy -- two distinct histories that hash alike collapse into one
//! infoset. Reach for it only when allocation is intolerable and a vanishingly small collision
//! probability is acceptable.
//!
//! [`Game::Infoset`]: crate::Game::Infoset

use crate::splitmix::mix;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::sync::Arc;

/// A persistent, append-only history usable as a [`Game::Infoset`](crate::Game::Infoset) key.
///
/// Internally a singly-linked stack of [`Arc`] nodes sharing their tails, so [`push`](Self::push)
/// and [`Clone`] are O(1) and never copy the existing history. Each node caches the rolling hash of
/// the history up to it, so [`Hash`] is O(1); [`Eq`] short-circuits on a shared tail or a digest
/// mismatch and otherwise compares structurally, so it is exact despite the cached hash.
///
/// `T` is the per-step action; it must be [`Hash`] to append and [`Eq`] to compare histories. The
/// [`Arc`] backing makes a `History` [`Send`] and [`Sync`] whenever `T` is, so it works as a
/// [`LazySolver`](crate::LazySolver) infoset under parallel solving.
pub struct History<T>(Option<Arc<Node<T>>>);

struct Node<T> {
    value: T,
    len: usize,
    digest: u64,
    rest: History<T>,
}

impl<T> History<T> {
    /// An empty history.
    #[must_use]
    pub const fn new() -> Self {
        History(None)
    }

    /// The number of actions appended so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.as_ref().map_or(0, |node| node.len)
    }

    /// Whether no actions have been appended.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The most recently appended action, or `None` if the history is empty.
    #[must_use]
    pub fn head(&self) -> Option<&T> {
        self.0.as_ref().map(|node| &node.value)
    }

    /// The cached rolling hash of the whole history (`0` when empty).
    #[must_use]
    fn digest(&self) -> u64 {
        self.0.as_ref().map_or(0, |node| node.digest)
    }

    /// The history with `value` appended in O(1), sharing this history as the new tail instead of
    /// copying it.
    #[must_use]
    pub fn push(&self, value: T) -> History<T>
    where
        T: Hash,
    {
        History(Some(Arc::new(Node {
            len: self.len() + 1,
            digest: mix(self.digest(), &value),
            value,
            rest: self.clone(),
        })))
    }

    /// The actions from most- to least-recently appended.
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter(self.0.as_deref())
    }
}

// clone bumps the shared `Arc`; the elements are never cloned, so no `T: Clone` bound
impl<T> Clone for History<T> {
    fn clone(&self) -> Self {
        History(self.0.clone())
    }
}

impl<T> Default for History<T> {
    fn default() -> Self {
        History::new()
    }
}

// hashing the cached digest is consistent with `Eq` and needs no `T: Hash` bound, since `push`
// already folded every action into the digest
impl<T> Hash for History<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.digest());
    }
}

impl<T: PartialEq> PartialEq for History<T> {
    fn eq(&self, other: &History<T>) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            // a shared tail or a digest/length mismatch settles it in O(1); otherwise fall back to a
            // structural compare so a digest collision can't conflate distinct histories
            (Some(left), Some(right)) => {
                Arc::ptr_eq(left, right)
                    || (left.digest == right.digest
                        && left.len == right.len
                        && left.value == right.value
                        && left.rest == right.rest)
            }
            _ => false,
        }
    }
}

impl<T: Eq> Eq for History<T> {}

impl<T: Debug> Debug for History<T> {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T> IntoIterator for &'a History<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

/// Iterator over a [`History`]'s actions, most- to least-recently appended.
pub struct Iter<'a, T>(Option<&'a Node<T>>);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let node = self.0?;
        self.0 = node.rest.0.as_deref();
        Some(&node.value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.0.map_or(0, |node| node.len);
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}
impl<T> FusedIterator for Iter<'_, T> {}

/// A fixed-size rolling hash of an append-only history, usable as a cheap, allocation-free
/// [`Game::Infoset`](crate::Game::Infoset) key.
///
/// Unlike [`History`] it keeps no structure -- only `N` independent 64-bit hash lanes -- so it is
/// [`Copy`] and every operation is O(N) with no heap allocation. The trade-off is that it is
/// *lossy*: two distinct histories whose digests collide become the same infoset, silently merging
/// their regret.
///
/// `N` picks that trade-off. Each lane is its own rolling hash, salted by its index, so a collision
/// needs all `N` lanes to coincide at once -- roughly a `2^(64*N)` chance. The default `Digest<1>`
/// is most compact but a collision grows likely as the distinct-history count approaches `2^32`;
/// `Digest<2>` pushes that to `2^64` (negligible at any feasible game size) and `Digest<4>` further
/// still, each lane costing 8 more bytes of key. Prefer any of them only for astronomically large
/// games where avoiding allocation matters more than the residual risk; otherwise use the exact
/// [`History`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Digest<const N: usize = 1>([u64; N]);

impl<const N: usize> Digest<N> {
    /// The digest of an empty history.
    #[must_use]
    pub fn new() -> Self {
        Digest([0; N])
    }

    /// The digest with `value` folded in. O(N) and allocation-free.
    #[must_use]
    pub fn push(self, value: &impl Hash) -> Digest<N> {
        let mut lanes = self.0;
        for (index, lane) in lanes.iter_mut().enumerate() {
            // salt each lane by its index so the lanes evolve independently
            *lane = mix(*lane, &(index, value));
        }
        Digest(lanes)
    }
}

// no `derive(Default)`: `[u64; N]` isn't `Default` for an arbitrary `N`, but `[0; N]` always is
impl<const N: usize> Default for Digest<N> {
    fn default() -> Self {
        Digest::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{Digest, History};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of(value: &impl Hash) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn history_push_is_structural() {
        let base = History::new().push('a').push('b');
        let left = base.push('c');
        let right = base.push('c');
        assert_eq!(left, right);
        assert_eq!(hash_of(&left), hash_of(&right));
        assert_eq!(left.len(), 3);
        assert_eq!(left.head(), Some(&'c'));
        assert_ne!(left, base.push('d'));
        assert_ne!(left, base);
    }

    #[test]
    fn history_clone_shares_tail() {
        // a clone is a shared tail, so it compares equal via the pointer fast path
        let history = History::new().push(1).push(2);
        let clone = history.clone();
        assert_eq!(history, clone);
        assert_eq!(history.iter().copied().collect::<Vec<_>>(), vec![2, 1]);
    }

    #[test]
    fn empty_history_is_default() {
        let empty: History<u8> = History::new();
        assert!(empty.is_empty());
        assert_eq!(empty, History::default());
        assert_eq!(empty.head(), None);
        assert_eq!(empty.iter().count(), 0);
    }

    #[test]
    fn digest_matches_equal_histories() {
        // context-free, so spell out the single lane (real infosets infer N from their type)
        let left = Digest::<1>::new().push(&1).push(&2);
        let right = Digest::<1>::new().push(&1).push(&2);
        assert_eq!(left, right);
        assert_ne!(left, Digest::<1>::new().push(&2).push(&1));
        assert_ne!(left, Digest::<1>::new());
    }

    #[test]
    fn digest_width_is_parametric() {
        let wide: Digest<2> = Digest::new().push(&1).push(&2);
        assert_eq!(wide, Digest::<2>::new().push(&1).push(&2));
        assert_ne!(wide, Digest::<2>::new().push(&2).push(&1));
        let [low, high] = wide.0; // white-box: the lanes are visible from the same module
        assert_ne!(low, high, "lanes should not be identical");
    }
}
