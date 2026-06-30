//! splitmix64 mixing primitives, shared by the deterministic sampler and the infoset-key hashers.

use std::hash::{Hash, Hasher};

/// The splitmix64 increment (the golden-ratio gamma constant).
pub(crate) const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// The splitmix64 finalizer: scramble a state word into a well-distributed output.
///
/// It is a bijection, so it loses no information -- hence very low collision odds when chained as a
/// hash step.
pub(crate) fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Fold `value` into a running splitmix64 `state`: mix the value in, then finalize.
pub(crate) fn fold(state: u64, value: u64) -> u64 {
    finalize((state ^ value).wrapping_add(GAMMA))
}

/// Advance `x` by one splitmix64 step (equivalent to [`fold`] with a zero value).
pub(crate) fn splitmix(x: u64) -> u64 {
    finalize(x.wrapping_add(GAMMA))
}

/// A cheap, non-cryptographic [`Hasher`] that folds each 64-bit word through [`fold`].
///
/// Infoset keys aren't adversarial, so the denial-of-service resistance of a cryptographic hash buys
/// nothing here; this keeps splitmix's strong avalanche (low collision odds, since it is a bijection)
/// at a fraction of the cost. Usable as a `BuildHasherDefault` for the regret table.
#[derive(Default)]
pub(crate) struct SplitmixHasher(u64);

impl Hasher for SplitmixHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.write_u64(u64::from_le_bytes(*word));
        }
        if !rest.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(tail));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = fold(self.0, value);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

/// Fold `value` into the running hash `seed` via [`SplitmixHasher`] -- the rolling-hash step behind the
/// [`History`](crate::History) and [`Digest`](crate::Digest) infoset keys.
pub(crate) fn mix(seed: u64, value: &impl Hash) -> u64 {
    let mut hasher = SplitmixHasher(seed);
    value.hash(&mut hasher);
    hasher.finish()
}
