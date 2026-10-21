pub mod bitsync;
pub mod demod;
pub mod frontend;
pub mod tone;
pub mod window;
pub mod charset;
pub mod combine;
pub mod params;
pub mod score;
pub mod softlock;

#[cfg(test)]
mod testutil;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
