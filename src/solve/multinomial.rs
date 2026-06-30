use rand::{Rng, RngExt};

/// Sample an action index from its probabilities by a single inverse-CDF pass.
///
/// Takes the probabilities as an iterator (not a slice) so the caller can stream them straight from
/// any layout -- e.g. a strided field of an array-of-structs -- without copying into a contiguous
/// buffer. The last probability is the implied remainder, so a draw that exhausts all the earlier
/// ones lands on it.
pub fn sample<R: Rng + ?Sized>(rng: &mut R, probs: impl ExactSizeIterator<Item = f32>) -> usize {
    let last = probs.len() - 1;
    let mut remaining: f32 = rng.random();
    for (index, prob) in probs.take(last).enumerate() {
        if prob < remaining {
            remaining -= prob;
        } else {
            return index;
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::sample;
    use rand::rng;

    #[test]
    fn test_invalid_small() {
        for _ in 0..1000 {
            assert!((0..3).contains(&sample(&mut rng(), [0.2, 0.3, 0.4].into_iter())));
        }
    }

    #[test]
    fn test_invalid_large() {
        for _ in 0..1000 {
            assert!((0..3).contains(&sample(&mut rng(), [0.2, 0.7, 0.4].into_iter())));
        }
    }
}
