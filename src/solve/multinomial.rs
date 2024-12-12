use rand::Rng;
use rand_distr::Distribution;

pub struct Multinomial<'a> {
    // We store all be the last, since that should sum to one
    init_probs: &'a [f64],
}

impl<'a> Multinomial<'a> {
    pub fn new(probs: &'a [f64]) -> Self {
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
        let mut remaining: f64 = rnd.gen();
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
    use rand::thread_rng;
    use rand_distr::Distribution;

    #[test]
    fn test_invalid_small() {
        let dist = Multinomial::new(&[0.2, 0.3, 0.4]);
        for _ in 0..1000 {
            assert!((0..3).contains(&dist.sample(&mut thread_rng())));
        }
    }

    #[test]
    fn test_invalid_large() {
        let dist = Multinomial::new(&[0.2, 0.7, 0.4]);
        for _ in 0..1000 {
            assert!((0..3).contains(&dist.sample(&mut thread_rng())));
        }
    }
}
