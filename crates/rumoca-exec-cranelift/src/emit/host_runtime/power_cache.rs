//! Exact reuse of finite native powers; no model or solver state is retained.

use std::cell::RefCell;

const INDEX_BITS: u32 = 12;
const CAPACITY: usize = 1 << INDEX_BITS;

#[derive(Clone, Copy)]
struct Power {
    x: u64,
    y: u64,
    value: f64,
}

struct PowerCache {
    entries: [Option<Power>; CAPACITY],
}

impl PowerCache {
    const fn new() -> Self {
        Self {
            entries: [None; CAPACITY],
        }
    }

    fn index(x: u64, y: u64) -> usize {
        ((x ^ y.rotate_left(23)).wrapping_mul(0x517c_c1b7_2722_0a95) >> (64 - INDEX_BITS)) as usize
    }

    fn powf(&mut self, x: f64, y: f64) -> f64 {
        // Exceptional inputs and results keep the ordinary math call, including
        // its NaN propagation. Finite hits preserve the complete input/result
        // bits; collisions only replace an entry and never authorize reuse.
        if !x.is_finite() || !y.is_finite() {
            return x.powf(y);
        }
        let (x_bits, y_bits) = (x.to_bits(), y.to_bits());
        let entry = &mut self.entries[Self::index(x_bits, y_bits)];
        if let Some(power) = entry
            && power.x == x_bits
            && power.y == y_bits
        {
            return power.value;
        }
        let value = x.powf(y);
        if value.is_finite() {
            *entry = Some(Power {
                x: x_bits,
                y: y_bits,
                value,
            });
        }
        value
    }
}

thread_local! {
    // One bounded direct-mapped table per executor thread. There is no model
    // identity, numerical history, allocation on a hit, or cross-thread order.
    static POWERS: RefCell<PowerCache> = const { RefCell::new(PowerCache::new()) };
}

pub(super) fn powf(x: f64, y: f64) -> f64 {
    POWERS.with(|cache| cache.borrow_mut().powf(x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_preserve_bits_and_collision_replacement() {
        let mut cache = PowerCache::new();
        for x in [
            0.0,
            -0.0,
            0.125,
            -0.125,
            1.0,
            -1.0,
            2.0,
            -2.0,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
        ] {
            for y in [-3.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.5] {
                let expected = x.powf(y).to_bits();
                assert_eq!(cache.powf(x, y).to_bits(), expected);
                assert_eq!(cache.powf(x, y).to_bits(), expected);
            }
        }
        let index = PowerCache::index(2.0_f64.to_bits(), 3.0_f64.to_bits());
        let collision = (3..100_000)
            .map(f64::from)
            .find(|x| PowerCache::index(x.to_bits(), 3.0_f64.to_bits()) == index)
            .expect("the bounded table has collisions");
        assert_eq!(cache.powf(2.0, 3.0).to_bits(), 8.0_f64.to_bits());
        assert_eq!(
            cache.powf(collision, 3.0).to_bits(),
            collision.powf(3.0).to_bits()
        );
        assert_eq!(cache.entries[index].unwrap().x, collision.to_bits());
        assert_eq!(cache.powf(2.0, 3.0).to_bits(), 8.0_f64.to_bits());
    }

    #[test]
    fn exceptional_inputs_and_results_are_not_retained() {
        let mut cache = PowerCache::new();
        for (x, y) in [
            (f64::NAN, 2.0),
            (2.0, f64::NAN),
            (f64::INFINITY, 2.0),
            (2.0, f64::NEG_INFINITY),
            (-2.0, 0.5),
            (0.0, -1.0),
            (f64::MAX, 2.0),
        ] {
            assert_eq!(cache.powf(x, y).to_bits(), x.powf(y).to_bits());
        }
        assert!(cache.entries.iter().all(Option::is_none));
    }
}
