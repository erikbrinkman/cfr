//! Split slices into uneven chunks
use std::iter::FusedIterator;
use std::mem;

pub struct SplitsBy<'a, T, I> {
    slice: &'a [T],
    lens: I,
}

impl<'a, T, I: Iterator<Item = usize>> Iterator for SplitsBy<'a, T, I> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        match self.lens.next() {
            Some(len) => {
                let (ret, rest) = self.slice.split_at(len);
                self.slice = rest;
                Some(ret)
            }
            None => None,
        }
    }
}

impl<T, I: FusedIterator<Item = usize>> FusedIterator for SplitsBy<'_, T, I> {}

pub fn split_by<T, I: Iterator<Item = usize>, N: IntoIterator<IntoIter = I>>(
    slice: &[T],
    lens: N,
) -> SplitsBy<'_, T, I> {
    SplitsBy {
        slice,
        lens: lens.into_iter(),
    }
}

pub struct SplitsByMut<'a, T, I> {
    slice: &'a mut [T],
    lens: I,
}

impl<'a, T, I: Iterator<Item = usize>> Iterator for SplitsByMut<'a, T, I> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        match self.lens.next() {
            Some(len) => {
                let tmp = mem::take(&mut self.slice);
                let (ret, rest) = tmp.split_at_mut(len);
                self.slice = rest;
                Some(ret)
            }
            None => None,
        }
    }
}

impl<T, I: FusedIterator<Item = usize>> FusedIterator for SplitsByMut<'_, T, I> {}

pub fn split_by_mut<T, I: Iterator<Item = usize>, N: IntoIterator<IntoIter = I>>(
    slice: &mut [T],
    lens: N,
) -> SplitsByMut<'_, T, I> {
    SplitsByMut {
        slice,
        lens: lens.into_iter(),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_ref() {
        let slices = [1, 2, 3, 4, 5, 6, 7, 8];
        let sliced: Box<[&[_]]> = super::split_by(&slices, [1, 3, 4]).collect();
        assert_eq!(*sliced, [vec![1], vec![2, 3, 4], vec![5, 6, 7, 8]]);
    }

    #[test]
    fn test_mut() {
        let mut slices = [1, 2, 3, 4, 5, 6, 7, 8];
        for mut_slice in super::split_by_mut(&mut slices, [1, 3, 4]) {
            mut_slice.fill(0);
        }
        assert_eq!(slices, [0; 8]);
    }
}
