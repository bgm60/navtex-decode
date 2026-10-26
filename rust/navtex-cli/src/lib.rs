//! Building blocks of the NAVTEX command-line program.
//!
//! The decoding itself lives in `navtex-core`. This crate adds what is
//! around it: the signal-quality meter, the assembly of characters into
//! lines, the outputs (console, text log, database), and the audio input
//! (WAV files so far; live audio and settings come in later steps).

pub mod console;
pub mod lines;
pub mod pipeline;
pub mod quality;
pub mod run;
pub mod source;
pub mod sqlitelog;
pub mod textlog;
pub mod utc;
pub mod wav;

#[cfg(test)]
mod testutil;
