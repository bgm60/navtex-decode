//! Python bindings for the NAVTEX decoder core.
//!
//! ```python
//! import navtex_rs
//! decoder = navtex_rs.Decoder()
//! for chunk in audio_chunks:                    # mono float32 arrays
//!     for ch, bit_index in decoder.push(chunk):
//!         print(ch, end="")
//! for ch, bit_index in decoder.finish():
//!     print(ch, end="")
//! ```

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymodule]
mod navtex_rs {
    use super::*;
    use navtex_core::frontend::FrontEndConfig;
    use navtex_core::params::SoftLockParams;

    /// The NAVTEX decoder: audio in, characters out.
    ///
    /// Feed it mono float32 audio with `push` in chunks of any size, and
    /// call `finish` at the end. The settings default to your profile's.
    #[pyclass]
    struct Decoder {
        inner: navtex_core::decoder::Decoder,
    }

    #[pymethods]
    impl Decoder {
        #[new]
        #[pyo3(signature = (sample_rate=48000, mark_freq=1785.0, space_freq=1615.0, oversample=8, loop_gain=0.05))]
        fn new(
            sample_rate: u32,
            mark_freq: f64,
            space_freq: f64,
            oversample: u32,
            loop_gain: f64,
        ) -> PyResult<Self> {
            let nyquist = f64::from(sample_rate) / 2.0;
            if sample_rate < 1000 {
                return Err(PyValueError::new_err("sample_rate must be at least 1000"));
            }
            if !(mark_freq > 0.0 && mark_freq < nyquist && space_freq > 0.0 && space_freq < nyquist) {
                return Err(PyValueError::new_err(
                    "mark_freq and space_freq must be above 0 and below half the sample rate",
                ));
            }
            if !(1..=64).contains(&oversample) {
                return Err(PyValueError::new_err("oversample must be between 1 and 64"));
            }
            if !(loop_gain.is_finite() && loop_gain >= 0.0) {
                return Err(PyValueError::new_err("loop_gain must be 0 or more"));
            }
            let config = FrontEndConfig { sample_rate, oversample, mark_freq, space_freq };
            Ok(Decoder {
                inner: navtex_core::decoder::Decoder::new(
                    &config,
                    loop_gain,
                    SoftLockParams::default(),
                ),
            })
        }

        /// Feeds in a chunk of mono float32 audio (a numpy array, or anything
        /// with the buffer protocol). Returns a list of (character,
        /// bit_index) pairs for the characters that became ready. Audio of
        /// another type, such as float64, raises BufferError: convert it
        /// first with `.astype(numpy.float32)`.
        fn push(&mut self, py: Python<'_>, samples: PyBuffer<f32>) -> PyResult<Vec<(String, usize)>> {
            let audio = samples.to_vec(py)?;
            // Decoding does not need Python, so let other Python threads run.
            let events = py.detach(|| self.inner.push(&audio));
            Ok(to_pairs(events))
        }

        /// Call at the end of the audio: returns what is still held back.
        fn finish(&mut self, py: Python<'_>) -> Vec<(String, usize)> {
            to_pairs(py.detach(|| self.inner.finish()))
        }

        /// The confidence (0 to 1) of every bit decision since the last
        /// call, which is then forgotten. Average them for signal strength.
        fn take_confidences(&mut self) -> Vec<f64> {
            self.inner.take_confidences()
        }

        /// "data" while a signal is locked, otherwise "searching".
        #[getter]
        fn state(&self) -> &'static str {
            if self.inner.is_locked() { "data" } else { "searching" }
        }

        /// How many times a lock has been released.
        #[getter]
        fn releases(&self) -> usize {
            self.inner.releases()
        }

        /// How many times the lock has moved to another alignment.
        #[getter]
        fn switches(&self) -> usize {
            self.inner.switches()
        }
    }

    fn to_pairs(events: Vec<navtex_core::softlock::LockEvent>) -> Vec<(String, usize)> {
        events.into_iter().map(|e| (e.ch.to_string(), e.bit_index)).collect()
    }
}
