//! The NAVTEX decoder command-line program.
//!
//! It decodes a WAV recording or live audio from a sound device. Settings come
//! from a profile in a TOML file and/or the command line, and the text goes to
//! the console, a text log and a database as the profile says.

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};
use navtex_cli::audio::{self, LiveSource};
use navtex_cli::config::{self, Mode, Profile};
use navtex_cli::console::ConsoleSink;
use navtex_cli::level::run_level;
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const AFTER_HELP: &str = "\
Settings come from a named profile in a TOML file (the same file the Python
program uses). Options given here override the profile.

Exit status: 0 on success; 1 if the audio or an output could not be read or
written (a recording that ends early is decoded as far as it goes, then
reported) or the audio input fails; 2 for a mistake on the command line or in
the settings. Ctrl+C stops decoding cleanly (exit status 0).";

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

    /// List the audio input devices (their numbers are what `device = N` means) and exit
    #[arg(long)]
    list_devices: bool,

    /// Show an input-level meter and the signal quality instead of decoding to the
    /// console; nothing is logged. Use it to check the device and its level
    #[arg(long)]
    level: bool,
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
    if cli.list_devices {
        return list_devices();
    }
    let profile = resolve(&cli);
    if cli.level && profile.quiet {
        usage_error("--level draws the meter on the console, so it cannot be used with quiet");
    }
    let quiet = profile.quiet;
    let say = |message: &str| {
        if !quiet {
            println!("{message}");
        }
    };

    // Ctrl+C asks the run to stop; the run ends the line, closes the logs
    // and returns. A second Ctrl+C, if the first did not work, ends the
    // program at once.
    let stop = Arc::new(AtomicBool::new(false));
    let handler_flag = Arc::clone(&stop);
    let handler = ctrlc::set_handler(move || {
        if handler_flag.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
    });
    if let Err(error) = handler {
        eprintln!("warning: Ctrl+C will not stop the program cleanly: {error}");
    }

    let rate = FrontEndConfig::default().sample_rate;
    let mut source: Box<dyn AudioSource> = match profile.mode {
        Mode::File => {
            let wav_file = profile.wav_file.as_deref().unwrap_or_default();
            match WavSource::open(wav_file, rate) {
                Ok(source) => {
                    say(&format!("Decoding file: {wav_file}"));
                    Box::new(source)
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    return ExitCode::from(1);
                }
            }
        }
        Mode::Live => {
            let warn = Box::new(|text: &str| eprintln!("warning: {text}"));
            match LiveSource::open(profile.device.as_ref(), rate, Arc::clone(&stop), warn) {
                Ok(source) => {
                    say(&format!("Listening on {}... (Ctrl+C to stop)", source.description()));
                    Box::new(source)
                }
                Err(error) => {
                    eprintln!("error: {error}");
                    return ExitCode::from(1);
                }
            }
        }
    };

    let config = FrontEndConfig {
        mark_freq: profile.mark_freq(),
        space_freq: profile.space_freq(),
        ..FrontEndConfig::default()
    };
    let window = profile.signal_quality_window as usize;
    let new_pipeline = |sinks: Vec<Box<dyn LineSink>>| {
        Pipeline::new(&config, profile.loop_gain, SoftLockParams::default(), window, LineAssembler::new(sinks))
    };

    let result = if cli.level {
        // No sinks: the decoder runs only to measure the signal quality.
        println!("Input level meter: nothing is logged.");
        run_level(source.as_mut(), &mut new_pipeline(Vec::new()), &stop, &mut std::io::stdout())
    } else {
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
        run(source.as_mut(), &mut new_pipeline(sinks), &stop)
    };
    // Close the sound card before saying anything more.
    drop(source);

    if let Err(error) = result {
        // The decoded text may stop part-way along a line, so start a new one.
        let gap = if quiet { "" } else { "\n" };
        eprintln!("{gap}error: {error}");
        return ExitCode::from(1);
    }
    if stop.load(Ordering::SeqCst) {
        say("\n\nStopped by user.");
    }
    ExitCode::SUCCESS
}

/// `--list-devices`: prints the input devices, numbered as `device = N` counts them.
fn list_devices() -> ExitCode {
    let rate = FrontEndConfig::default().sample_rate;
    match audio::list_devices(rate) {
        Ok(devices) if devices.is_empty() => {
            eprintln!("error: no audio input devices found ({})", audio::host_name());
            ExitCode::from(1)
        }
        Ok(devices) => {
            println!("Audio input devices ({}):", audio::host_name());
            for (index, device) in devices.iter().enumerate() {
                let mut notes = Vec::new();
                if device.is_default {
                    notes.push("default".to_string());
                }
                if !device.usable {
                    notes.push(format!("cannot record at {rate} Hz"));
                }
                let notes = if notes.is_empty() { String::new() } else { format!("  [{}]", notes.join(", ")) };
                println!("  {index}: {}{notes}", device.name);
            }
            println!("Use the number, or part of the name, as device = ... in a profile.");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
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
