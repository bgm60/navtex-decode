//! The NAVTEX decoder command-line program.
//!
//! Work in progress: it decodes one WAV recording, settings come from a
//! profile in a TOML file and/or the command line, and the text goes to the
//! console, a text log and a database as the profile says. Live audio comes
//! in the next step.

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};
use navtex_cli::config::{self, Mode, Profile};
use navtex_cli::console::ConsoleSink;
use navtex_cli::lines::{LineAssembler, LineSink};
use navtex_cli::pipeline::Pipeline;
use navtex_cli::run::run;
use navtex_cli::source::AudioSource;
use navtex_cli::sqlitelog::SqliteLogSink;
use navtex_cli::textlog::TextLogSink;
use navtex_cli::wav::WavSource;
use navtex_core::frontend::FrontEndConfig;
use navtex_core::params::SoftLockParams;
use std::path::Path;
use std::process::ExitCode;

const AFTER_HELP: &str = "\
Settings come from a named profile in a TOML file (the same file the Python
program uses). Options given here override the profile.

Exit status: 0 on success; 1 if the audio or an output could not be read or
written (a recording that ends early is decoded as far as it goes, then
reported); 2 for a mistake on the command line or in the settings.";

/// Decodes NAVTEX (SITOR-B, 100 baud FSK) transmissions.
#[derive(Parser)]
#[command(name = "navtex-cli", version, after_help = AFTER_HELP)]
struct Cli {
    /// Name of the profile (a [section]) to use from the settings file
    profile: Option<String>,

    /// The settings file
    #[arg(long, value_name = "PATH", default_value = "navtex.toml")]
    config: String,

    /// Decode this 48000 Hz WAV recording (sets mode = "file"); with no
    /// profile, the built-in defaults are used
    #[arg(long, value_name = "FILE")]
    wav: Option<String>,

    /// Print no status messages and no decoded text (needs log_dir or
    /// db_file); errors and warnings are still shown
    #[arg(short, long, conflicts_with = "no_quiet")]
    quiet: bool,

    /// Print status messages and text, even if the profile says quiet = true
    #[arg(long)]
    no_quiet: bool,
}

/// Reports a mistake in the arguments or settings the way clap does and
/// exits with status 2.
fn usage_error(message: impl std::fmt::Display) -> ! {
    Cli::command().error(ErrorKind::ValueValidation, message).exit()
}

/// The settings to use: profile first, then the command-line options on top.
fn resolve(cli: &Cli) -> Profile {
    let mut profile = match &cli.profile {
        Some(name) => match config::load_profile(&cli.config, name) {
            Ok(loaded) => {
                for warning in &loaded.warnings {
                    eprintln!("warning: {warning}");
                }
                loaded.profile
            }
            Err(error) => usage_error(error),
        },
        None if cli.wav.is_some() => Profile::default(),
        None => Cli::command()
            .error(ErrorKind::MissingRequiredArgument, "give a profile name, or --wav FILE")
            .exit(),
    };
    if let Some(path) = &cli.wav {
        profile.mode = Mode::File;
        profile.wav_file = Some(path.clone());
    }
    if cli.quiet {
        profile.quiet = true;
    }
    if cli.no_quiet {
        profile.quiet = false;
    }
    if let Err(error) = profile.validate() {
        usage_error(error);
    }
    profile
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let profile = resolve(&cli);
    let quiet = profile.quiet;
    let say = |message: &str| {
        if !quiet {
            println!("{message}");
        }
    };

    if profile.mode == Mode::Live {
        usage_error("live audio is not available yet: this version decodes WAV files only (use --wav FILE)");
    }
    let wav_file = profile.wav_file.as_deref().unwrap_or_default();
    let rate = FrontEndConfig::default().sample_rate;
    let mut source = match WavSource::open(wav_file, rate) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(1);
        }
    };
    say(&format!("Decoding file: {wav_file}"));

    let mut sinks: Vec<Box<dyn LineSink>> = Vec::new();
    if !quiet {
        sinks.push(Box::new(ConsoleSink::stdout()));
    }
    if let Err(error) = open_logs(&profile, &source.description(), &mut sinks, &say) {
        // Close what was opened, so no log is left half-written.
        for sink in &mut sinks {
            if let Err(e) = sink.close() {
                eprintln!("warning: {e}");
            }
        }
        eprintln!("error: could not open log output: {error}");
        return ExitCode::from(1);
    }

    let config = FrontEndConfig {
        mark_freq: profile.mark_freq(),
        space_freq: profile.space_freq(),
        ..FrontEndConfig::default()
    };
    let window = profile.signal_quality_window as usize;
    let mut pipeline = Pipeline::new(
        &config,
        profile.loop_gain,
        SoftLockParams::default(),
        window,
        LineAssembler::new(sinks),
    );
    match run(&mut source, &mut pipeline) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // The decoded text may stop part-way along a line, so start a new one.
            let gap = if quiet { "" } else { "\n" };
            eprintln!("{gap}error: {error}");
            ExitCode::from(1)
        }
    }
}

/// Opens the text log and the database that the profile asks for.
fn open_logs(
    profile: &Profile,
    source: &str,
    sinks: &mut Vec<Box<dyn LineSink>>,
    say: &dyn Fn(&str),
) -> Result<(), String> {
    if let Some(dir) = profile.log_dir.as_deref().filter(|s| !s.is_empty()) {
        let log = TextLogSink::create(Path::new(dir), source).map_err(|e| format!("{dir}: {e}"))?;
        say(&format!("Logging decoded text to: {}", log.path().display()));
        sinks.push(Box::new(log));
    }
    if let Some(file) = profile.db_file.as_deref().filter(|s| !s.is_empty()) {
        let db = SqliteLogSink::create(Path::new(file))?;
        say(&format!("Logging decoded lines to database: {}", db.path().display()));
        sinks.push(Box::new(db));
    }
    Ok(())
}
