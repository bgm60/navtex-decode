//! The console output: decoded text written to the screen as it arrives.

use crate::lines::{LineSink, SinkResult};
use std::io::{self, Write};

/// Writes every character exactly as decoded (CR and LF included) and
/// flushes at once, so text appears as it is received.
///
/// Rust does not translate line endings on Windows, so unlike the Python
/// version nothing special is needed to stop a CR LF becoming CR CR LF.
pub struct ConsoleSink {
    out: Box<dyn Write>,
}

impl ConsoleSink {
    /// An output to the program's standard output.
    pub fn stdout() -> Self {
        ConsoleSink::new(io::stdout())
    }

    /// An output to anything that can be written to (tests use a memory buffer).
    pub fn new(out: impl Write + 'static) -> Self {
        ConsoleSink { out: Box::new(out) }
    }
}

impl LineSink for ConsoleSink {
    fn name(&self) -> &str {
        "ConsoleSink"
    }

    fn write(&mut self, ch: char) -> SinkResult {
        let mut buffer = [0u8; 4];
        self.out.write_all(ch.encode_utf8(&mut buffer).as_bytes()).map_err(|e| e.to_string())?;
        self.out.flush().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lines::LineAssembler;
    use crate::pipeline::Pipeline;
    use crate::testutil::{self, SharedBuf};

    #[test]
    fn text_comes_out_exactly_as_it_went_in() {
        let buf = SharedBuf::new();
        let mut assembler = LineAssembler::new(vec![Box::new(ConsoleSink::new(buf.clone()))]);
        for ch in "ZCZC KA91\r\nTEXT\r\n".chars() {
            assembler.feed(ch, || 50);
        }
        assert_eq!(buf.contents(), b"ZCZC KA91\r\nTEXT\r\n".to_vec());
    }

    #[test]
    fn a_failing_stream_is_reported_and_the_output_dropped() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("closed"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut sink = ConsoleSink::new(Broken);
        assert_eq!(sink.write('A'), Err("closed".to_string()));

        // Inside an assembler, the failure is warned about once and the
        // output is dropped; decoding carries on.
        let warnings = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
        let collected = std::rc::Rc::clone(&warnings);
        let mut assembler = LineAssembler::new(vec![Box::new(ConsoleSink::new(Broken))])
            .with_warn(move |message| collected.borrow_mut().push(message.to_string()));
        for ch in "AB".chars() {
            assembler.feed(ch, || 0);
        }
        assert_eq!(warnings.borrow().len(), 1);
        assert!(warnings.borrow()[0].contains("ConsoleSink failed (closed)"));
    }

    /// The real decode of a recording, sent to the console output, must give
    /// the same characters that Python sent to its outputs.
    #[test]
    fn a_whole_recording_reaches_the_console_unchanged() {
        for set in ["real_clip1", "scen_two_messages"] {
            let samples = testutil::read_wav(&testutil::vector_path(set, "input.wav"));
            let buf = SharedBuf::new();
            let mut pipeline = Pipeline::with_defaults(LineAssembler::new(vec![Box::new(
                ConsoleSink::new(buf.clone()),
            )]));
            for chunk in samples.chunks(4096) {
                pipeline.push(chunk);
            }
            pipeline.finish();

            let want: String = testutil::read_line_events(set)
                .iter()
                .filter_map(|row| row.strip_prefix("W,"))
                .map(|code| char::from_u32(code.parse().unwrap()).unwrap())
                .collect();
            assert!(want.len() > 100, "{set}: no text decoded");
            assert_eq!(String::from_utf8(buf.contents()).unwrap(), want, "{set}");
        }
    }
}
