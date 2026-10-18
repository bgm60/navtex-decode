use pyo3::prelude::*;

/// Python bindings for the NAVTEX decoder core.
#[pymodule]
mod navtex_rs {
    use pyo3::prelude::*;

    /// Temporary function to prove the two-crate setup works.
    #[pyfunction]
    fn add(a: u64, b: u64) -> u64 {
        navtex_core::add(a, b)
    }
}