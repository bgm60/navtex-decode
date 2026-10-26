//! Running a decode: pull audio from a source and push it through the pipeline.

use crate::pipeline::Pipeline;
use crate::source::AudioSource;

/// Decodes everything the source provides.
///
/// The pipeline is always finished, so the characters still being held back
/// come out, the last line ends and the outputs are closed, even when the
/// source fails part-way. In that case the error is returned afterwards:
/// whatever could be decoded has been, and the caller reports the problem.
pub fn run(source: &mut dyn AudioSource, pipeline: &mut Pipeline) -> Result<(), String> {
    let result = loop {
        match source.next_chunk() {
            Ok(Some(chunk)) => pipeline.push(&chunk),
            Ok(None) => break Ok(()),
            Err(error) => break Err(error),
        }
    };
    pipeline.finish();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lines::LineAssembler;
    use crate::testutil::{self, Recorder};
    use crate::wav::WavSource;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    fn pipeline(rows: &Rc<RefCell<Vec<String>>>) -> Pipeline {
        Pipeline::with_defaults(LineAssembler::new(vec![Box::new(Recorder::new(rows))]))
    }

    /// Decodes a recording through the real WAV reader and returns the line
    /// events without the final "close" row.
    fn decode_file(path: &str) -> Vec<String> {
        let rows = Rc::new(RefCell::new(Vec::new()));
        let mut source = WavSource::open(path, 48_000).expect("cannot open the recording");
        run(&mut source, &mut pipeline(&rows)).expect("decoding failed");
        let mut rows = rows.borrow().clone();
        assert_eq!(rows.pop().as_deref(), Some("close"));
        rows
    }

    /// The same check as `line_events_match_python_for_every_recording`, but
    /// starting from the file on disk and going through the WAV reader.
    #[test]
    fn every_recording_decodes_through_the_wav_reader_as_python_does() {
        let sets = testutil::sets_with("lines.csv");
        assert!(sets.len() >= 10, "too few sets: {}", sets.len());
        for set in sets {
            let got = decode_file(&testutil::vector_path(&set, "input.wav"));
            let want = testutil::read_line_events(&set);
            assert_eq!(got.len(), want.len(), "{set}: number of events");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{set}: event {k}");
            }
            println!("{set}: {} events identical to Python, read from the WAV file", want.len());
        }
    }

    /// A source that hands out ready-made chunks and then an error.
    struct FailingSource {
        chunks: VecDeque<Vec<f32>>,
        error: Option<String>,
    }

    impl AudioSource for FailingSource {
        fn description(&self) -> String {
            "test".to_string()
        }
        fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, String> {
            match self.chunks.pop_front() {
                Some(chunk) => Ok(Some(chunk)),
                None => match self.error.take() {
                    Some(error) => Err(error),
                    None => Ok(None),
                },
            }
        }
    }

    #[test]
    fn a_failing_source_still_flushes_and_closes_and_the_error_is_returned() {
        let samples = testutil::read_wav(&testutil::vector_path("real_clip1", "input.wav"));
        let rows = Rc::new(RefCell::new(Vec::new()));
        let mut source = FailingSource {
            chunks: samples[..samples.len() / 2].chunks(4096).map(<[f32]>::to_vec).collect(),
            error: Some("the file ended early".to_string()),
        };
        let result = run(&mut source, &mut pipeline(&rows));
        assert_eq!(result, Err("the file ended early".to_string()));
        let rows = rows.borrow();
        assert!(rows.len() > 50, "the text decoded before the failure should be kept");
        assert_eq!(rows.last().map(String::as_str), Some("close"), "the outputs should be closed");
        assert!(rows.iter().any(|row| row == "E"), "the last line should be ended");
    }

    #[test]
    fn an_empty_source_decodes_nothing_and_closes() {
        let rows = Rc::new(RefCell::new(Vec::new()));
        let mut source = FailingSource { chunks: VecDeque::new(), error: None };
        assert_eq!(run(&mut source, &mut pipeline(&rows)), Ok(()));
        assert_eq!(*rows.borrow(), vec!["close".to_string()]);
    }
}
