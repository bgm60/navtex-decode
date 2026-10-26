//! The NAVTEX decoder command-line program.
//!
//! Work in progress: for now it decodes one WAV recording to the console.
//! The settings file, logging to a file and a database, and live audio come
//! in later steps.

use navtex_cli::console::ConsoleSink;
use navtex_cli::lines::LineAssembler;
use navtex_cli::pipeline::Pipeline;
use navtex_cli::run::run;
use navtex_cli::wav::WavSource;
use navtex_core::frontend::FrontEndConfig;
use std::process::ExitCode;

const USAGE: &str = "usage: navtex-cli FILE.wav";

const HELP: &str = "\
usage: navtex-cli FILE.wav

Decodes a recording of a NAVTEX (SITOR-B) transmission and prints the text.

The recording must be a WAV file at 48000 Hz holding 8, 16, 24 or 32-bit
integer or 32-bit float samples; several channels are mixed to one.

  -h, --help    show this help and exit

Exit status: 0 if the whole file was decoded, 1 if it could not be read (or
ended early, after decoding what was there), 2 for a mistake on the command line.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "-h" || flag == "--help" => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
        [path] if !path.starts_with('-') => decode_file(path),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn decode_file(path: &str) -> ExitCode {
    let mut source = match WavSource::open(path, FrontEndConfig::default().sample_rate) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    println!("Decoding file: {path}");
    let mut pipeline = Pipeline::with_defaults(LineAssembler::new(vec![Box::new(ConsoleSink::stdout())]));
    match run(&mut source, &mut pipeline) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // The decoded text may stop part-way along a line, so start a new one.
            eprintln!("\nerror: {error}");
            ExitCode::from(1)
        }
    }
}
