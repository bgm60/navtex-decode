//! Where audio comes from.

/// A source of audio: a WAV file now, a sound card later. The rest of the
/// program asks for chunks one after another and does not care which kind
/// it has.
///
/// Audio is mono `f32` at the decoder's sample rate (48 kHz).
pub trait AudioSource {
    /// Says what is being decoded, for the log file header.
    fn description(&self) -> String;

    /// The next chunk of samples, any length. `Ok(None)` means the audio has
    /// ended. After an `Err` the source is finished, but the chunks already
    /// returned are valid: the caller should still decode them and flush.
    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, String>;
}
