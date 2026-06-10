use rand::distr::Distribution;
use rand::{Rng, RngExt};

#[derive(Debug)]
pub struct Multinomial<'a> {
    // We store all be the last, since that should sum to one
    init_probs: &'a [f32],
}

impl<'a> Multinomial<'a> {
    pub fn new(probs: &'a [f32]) -> Self {
        Multinomial {
            init_probs: &probs[..probs.len() - 1],
        }
    }
}

impl Distribution<usize> for Multinomial<'_> {
    fn sample<R>(&self, rnd: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        let mut remaining: f32 = rnd.random();
        let mut res = 0;
        for val in self.init_probs {
            if val < &remaining {
                remaining -= val;
                res += 1;
            } else {
                break;
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::Multinomial;
    use rand::distr::Distribution;
    use rand::rng;

    #[test]
    fn test_invalid_small() {
        let dist = Multinomial::new(&[0.2, 0.3, 0.4]);
        for _ in 0..1000 {
            assert!((0..3).contains(&dist.sample(&mut rng())));
        }
    }

    #[test]
    fn test_invalid_large() {
        let dist = Multinomial::new(&[0.2, 0.7, 0.4]);
        for _ in 0..1000 {
            assert!((0..3).contains(&dist.sample(&mut rng())));
        }
    }
}
