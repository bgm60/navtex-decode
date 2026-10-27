//! Live audio input from a sound device, using the `cpal` library (WASAPI on
//! Windows).
//!
//! The sound card delivers audio on a thread of its own, through a callback
//! that must return quickly. The callback therefore only converts the audio
//! to mono `f32` and puts it on a channel; [`LiveSource::next_chunk`] takes
//! it off again on the main thread. That keeps everything that is not about
//! the sound card (waiting, stopping, errors) testable without one: the
//! tests make a [`LiveSource`] from a channel they fill themselves.

use crate::config::Device;
use crate::source::AudioSource;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

/// How often a source that is waiting for audio looks at the stop flag.
const POLL: Duration = Duration::from_millis(100);

/// What the audio thread tells the main thread.
#[derive(Debug, PartialEq)]
pub enum Message {
    /// The next block of mono audio.
    Audio(Vec<f32>),
    /// Something went wrong that does not stop the decoding.
    Warning(String),
    /// The audio has stopped for good (the device was unplugged, say).
    Failed(String),
}

// ---------------------------------------------------------------------------
// Choosing a device
// ---------------------------------------------------------------------------

/// Which of the input devices to use.
///
/// * nothing asked for: the system's default input device;
/// * a number (or text that is only a number, as in the Python program):
///   the position in the `--list-devices` list;
/// * other text: the one device whose name contains it, ignoring case. If
///   several do, a name that matches exactly wins; otherwise it is an error
///   that lists them.
pub fn choose_device(names: &[String], default: Option<usize>, wanted: Option<&Device>) -> Result<usize, String> {
    let wanted = match wanted {
        None => {
            return default.ok_or_else(|| {
                "there is no default audio input device (see --list-devices, and choose one with device = ...)"
                    .to_string()
            });
        }
        Some(Device::Index(i)) => return by_index(names.len(), *i),
        Some(Device::Name(text)) => text.trim(),
    };
    if let Ok(i) = wanted.parse::<i64>() {
        return by_index(names.len(), i);
    }
    let lower = wanted.to_lowercase();
    let matches: Vec<usize> = (0..names.len()).filter(|&i| names[i].to_lowercase().contains(&lower)).collect();
    match matches.as_slice() {
        [] => Err(format!("no audio input device has {wanted:?} in its name (see --list-devices)")),
        [only] => Ok(*only),
        several => {
            let exact: Vec<usize> = several.iter().copied().filter(|&i| names[i].to_lowercase() == lower).collect();
            if let [one] = exact.as_slice() {
                return Ok(*one);
            }
            let list: Vec<String> = several.iter().map(|&i| format!("{i}: {}", names[i])).collect();
            Err(format!(
                "{wanted:?} matches more than one audio input device ({}); \
                 use more of the name, or the number",
                list.join("; ")
            ))
        }
    }
}

fn by_index(count: usize, index: i64) -> Result<usize, String> {
    match usize::try_from(index) {
        Ok(i) if i < count => Ok(i),
        _ if count == 0 => Err(format!("device {index} does not exist: there are no audio input devices")),
        _ => Err(format!(
            "device {index} does not exist: the input devices are numbered 0 to {} (see --list-devices)",
            count - 1
        )),
    }
}

// ---------------------------------------------------------------------------
// Turning what the sound card delivers into mono f32
// ---------------------------------------------------------------------------

/// Converts interleaved samples of any type to `f32` and averages the
/// channels of each frame into `out`, in the same way (and the same
/// arithmetic) as the WAV reader does.
pub fn mix_to_mono<T: Copy>(data: &[T], channels: usize, convert: impl Fn(T) -> f32, out: &mut Vec<f32>) {
    let channels = channels.max(1);
    out.clear();
    out.reserve(data.len() / channels);
    for frame in data.chunks_exact(channels) {
        let mut sum = convert(frame[0]);
        for &sample in &frame[1..] {
            sum += convert(sample);
        }
        out.push(if channels == 1 { sum } else { sum / channels as f32 });
    }
}

// ---------------------------------------------------------------------------
// The source
// ---------------------------------------------------------------------------

/// Live audio. Ends (returns `Ok(None)`) when the stop flag is set.
pub struct LiveSource {
    description: String,
    messages: Receiver<Message>,
    stop: Arc<AtomicBool>,
    warn: Box<dyn Fn(&str)>,
    /// Keeps the sound card running; dropping it stops the capture.
    _stream: Option<cpal::Stream>,
}

impl LiveSource {
    /// A source fed from a channel, for tests.
    pub fn from_channel(
        description: &str,
        messages: Receiver<Message>,
        stop: Arc<AtomicBool>,
        warn: Box<dyn Fn(&str)>,
    ) -> Self {
        LiveSource { description: description.to_string(), messages, stop, warn, _stream: None }
    }

    /// Opens the chosen input device at `rate` Hz and starts capturing.
    pub fn open(
        wanted: Option<&Device>,
        rate: u32,
        stop: Arc<AtomicBool>,
        warn: Box<dyn Fn(&str)>,
    ) -> Result<Self, String> {
        let devices = InputDevices::find()?;
        let names: Vec<String> = devices.list.iter().map(|d| d.name.clone()).collect();
        let default = devices.list.iter().position(|d| d.is_default);
        let index = choose_device(&names, default, wanted)?;
        let device = &devices.list[index];
        let (sender, receiver) = mpsc::channel();
        let stream = build_stream(&device.device, &device.name, rate, sender)?;
        stream.play().map_err(|e| format!("cannot start audio input '{}': {e}", device.name))?;
        Ok(LiveSource {
            description: format!("audio input {index}: {}", device.name),
            messages: receiver,
            stop,
            warn,
            _stream: Some(stream),
        })
    }
}

impl AudioSource for LiveSource {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, String> {
        loop {
            if self.stop.load(Ordering::SeqCst) {
                return Ok(None);
            }
            match self.messages.recv_timeout(POLL) {
                Ok(Message::Audio(samples)) => return Ok(Some(samples)),
                Ok(Message::Warning(text)) => (self.warn)(&text),
                Ok(Message::Failed(text)) => return Err(text),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("the audio input stopped unexpectedly".to_string());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Talking to the sound system (not testable without a sound card)
// ---------------------------------------------------------------------------

/// One input device, as shown by `--list-devices`.
pub struct InputInfo {
    pub name: String,
    pub is_default: bool,
    /// Whether it can capture at the decoder's sample rate.
    pub usable: bool,
}

struct FoundDevice {
    device: cpal::Device,
    name: String,
    is_default: bool,
}

struct InputDevices {
    list: Vec<FoundDevice>,
}

impl InputDevices {
    fn find() -> Result<Self, String> {
        let host = cpal::default_host();
        let default_id = host.default_input_device().and_then(|d| d.id().ok());
        let devices = host.input_devices().map_err(|e| format!("cannot list the audio input devices: {e}"))?;
        let list = devices
            .map(|device| {
                let name = device.description().map(|d| d.name().to_string()).unwrap_or_else(|_| {
                    device.id().map_or_else(|_| "(unnamed device)".to_string(), |id| id.to_string())
                });
                let is_default = default_id.is_some() && device.id().ok() == default_id;
                FoundDevice { device, name, is_default }
            })
            .collect();
        Ok(InputDevices { list })
    }
}

/// The name of the audio system in use, for `--list-devices`.
pub fn host_name() -> String {
    cpal::default_host().id().name().to_string()
}

/// Every input device, in the order that `device = N` counts them.
pub fn list_devices(rate: u32) -> Result<Vec<InputInfo>, String> {
    Ok(InputDevices::find()?
        .list
        .iter()
        .map(|d| InputInfo { name: d.name.clone(), is_default: d.is_default, usable: pick_config(&d.device, rate).is_some() })
        .collect())
}

/// How good a sample format is for us: lower is better, `None` is unusable.
fn format_rank(format: SampleFormat) -> Option<u8> {
    match format {
        SampleFormat::F32 => Some(0),
        SampleFormat::I32 => Some(1),
        SampleFormat::I16 => Some(2),
        SampleFormat::U16 => Some(3),
        SampleFormat::I8 => Some(4),
        _ => None,
    }
}

/// The device's configuration for `rate` Hz with the fewest channels and the
/// best sample format, if it has one.
fn pick_config(device: &cpal::Device, rate: u32) -> Option<cpal::SupportedStreamConfig> {
    device
        .supported_input_configs()
        .ok()?
        .filter(|range| range.contains_rate(rate))
        .filter_map(|range| format_rank(range.sample_format()).map(|rank| (range.channels(), rank, range)))
        .min_by_key(|&(channels, rank, _)| (channels, rank))
        .and_then(|(_, _, range)| range.try_with_sample_rate(rate))
}

fn build_stream(device: &cpal::Device, name: &str, rate: u32, sender: Sender<Message>) -> Result<cpal::Stream, String> {
    let config = pick_config(device, rate).ok_or_else(|| {
        format!(
            "the audio input '{name}' cannot record at {rate} Hz \
             (resampling is not implemented; set the device to {rate} Hz in the Windows sound settings)"
        )
    })?;
    let channels = usize::from(config.channels());
    let stream_config = config.config();
    let result = match config.sample_format() {
        SampleFormat::F32 => build::<f32>(device, stream_config, channels, sender),
        SampleFormat::I32 => build::<i32>(device, stream_config, channels, sender),
        SampleFormat::I16 => build::<i16>(device, stream_config, channels, sender),
        SampleFormat::U16 => build::<u16>(device, stream_config, channels, sender),
        SampleFormat::I8 => build::<i8>(device, stream_config, channels, sender),
        other => return Err(format!("the audio input '{name}' uses the sample format {other}, which is not supported")),
    };
    result.map_err(|e| format!("cannot open the audio input '{name}': {e}"))
}

/// Builds the input stream for one sample type.
fn build<T>(device: &cpal::Device, config: cpal::StreamConfig, channels: usize, sender: Sender<Message>) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + Copy,
    f32: FromSample<T>,
{
    let errors = sender.clone();
    let mut mono = Vec::new();
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            mix_to_mono(data, channels, f32::from_sample, &mut mono);
            // If the receiver has gone the program is ending, so a failed send is fine.
            let _ = sender.send(Message::Audio(mono.clone()));
        },
        move |error: cpal::Error| {
            let _ = errors.send(classify(&error));
        },
        None,
    )
}

/// Decides whether a stream error stops the decoding. Hiccups are warnings;
/// anything else, including the device being unplugged, is fatal.
fn classify(error: &cpal::Error) -> Message {
    use cpal::ErrorKind::{DeviceChanged, RealtimeDenied, Xrun};
    match error.kind() {
        Xrun => Message::Warning("audio buffer overrun: some audio may have been lost".to_string()),
        DeviceChanged | RealtimeDenied => Message::Warning(error.to_string()),
        _ => Message::Failed(format!("the audio input failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::thread;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn devices() -> Vec<String> {
        names(&["Microphone (USB Audio CODEC)", "Line In (Realtek HD Audio)", "Stereo Mix (Realtek HD Audio)"])
    }

    fn name(text: &str) -> Device {
        Device::Name(text.to_string())
    }

    // --- choose_device -----------------------------------------------------

    #[test]
    fn no_choice_means_the_default_device() {
        assert_eq!(choose_device(&devices(), Some(1), None), Ok(1));
        let error = choose_device(&devices(), None, None).unwrap_err();
        assert!(error.contains("no default audio input device"), "{error}");
    }

    #[test]
    fn a_number_picks_a_position_in_the_list() {
        assert_eq!(choose_device(&devices(), None, Some(&Device::Index(0))), Ok(0));
        assert_eq!(choose_device(&devices(), None, Some(&Device::Index(2))), Ok(2));
        let error = choose_device(&devices(), None, Some(&Device::Index(3))).unwrap_err();
        assert_eq!(error, "device 3 does not exist: the input devices are numbered 0 to 2 (see --list-devices)");
        assert!(choose_device(&devices(), None, Some(&Device::Index(-1))).is_err());
        let error = choose_device(&[], None, Some(&Device::Index(0))).unwrap_err();
        assert_eq!(error, "device 0 does not exist: there are no audio input devices");
    }

    #[test]
    fn text_that_is_only_a_number_is_a_position_as_in_python() {
        assert_eq!(choose_device(&devices(), None, Some(&name("2"))), Ok(2));
        assert_eq!(choose_device(&devices(), None, Some(&name(" 1 "))), Ok(1));
        assert!(choose_device(&devices(), None, Some(&name("7"))).unwrap_err().contains("does not exist"));
    }

    #[test]
    fn part_of_a_name_matches_ignoring_case() {
        assert_eq!(choose_device(&devices(), None, Some(&name("usb audio"))), Ok(0));
        assert_eq!(choose_device(&devices(), None, Some(&name("LINE IN"))), Ok(1));
        assert_eq!(choose_device(&devices(), None, Some(&name("stereo"))), Ok(2));
    }

    #[test]
    fn a_name_that_matches_nothing_is_an_error() {
        let error = choose_device(&devices(), None, Some(&name("bluetooth"))).unwrap_err();
        assert_eq!(error, "no audio input device has \"bluetooth\" in its name (see --list-devices)");
    }

    #[test]
    fn a_name_that_matches_several_is_an_error_that_lists_them() {
        let error = choose_device(&devices(), None, Some(&name("realtek"))).unwrap_err();
        assert!(error.contains("matches more than one"), "{error}");
        assert!(error.contains("1: Line In (Realtek HD Audio)") && error.contains("2: Stereo Mix"), "{error}");
        assert!(!error.contains("USB"), "{error}");
    }

    #[test]
    fn an_exact_name_wins_over_longer_names_that_contain_it() {
        let list = names(&["USB Audio", "USB Audio #2"]);
        assert_eq!(choose_device(&list, None, Some(&name("usb audio"))), Ok(0));
        assert!(choose_device(&list, None, Some(&name("usb"))).is_err());
    }

    // --- mix_to_mono ---------------------------------------------------------

    #[test]
    fn mono_audio_passes_through_unchanged() {
        let mut out = Vec::new();
        mix_to_mono(&[0.25_f32, -0.5, 1.0], 1, |x| x, &mut out);
        assert_eq!(out, [0.25, -0.5, 1.0]);
    }

    #[test]
    fn channels_are_averaged_frame_by_frame() {
        let mut out = Vec::new();
        mix_to_mono(&[1.0_f32, 0.0, 0.5, 0.5, -1.0, 1.0], 2, |x| x, &mut out);
        assert_eq!(out, [0.5, 0.5, 0.0]);
        mix_to_mono(&[0.3_f32, 0.6, 0.9], 3, |x| x, &mut out);
        assert_eq!(out, [(0.3_f32 + 0.6 + 0.9) / 3.0], "the sum is made in f32, like the WAV reader's");
    }

    #[test]
    fn a_partial_frame_at_the_end_is_dropped() {
        let mut out = Vec::new();
        mix_to_mono(&[1.0_f32, 1.0, 1.0, 1.0, 9.0], 2, |x| x, &mut out);
        assert_eq!(out, [1.0, 1.0]);
    }

    #[test]
    fn integer_samples_are_scaled_like_the_wav_reader_scales_them() {
        let mut out = Vec::new();
        mix_to_mono(&[i16::MIN, 16384, i16::MAX], 1, f32::from_sample, &mut out);
        assert_eq!(out, [-1.0, 0.5, i16::MAX as f32 / 32768.0]);
        mix_to_mono(&[0_u16, 32768, 49152], 1, f32::from_sample, &mut out);
        assert_eq!(out, [-1.0, 0.0, 0.5]);
    }

    // --- LiveSource ----------------------------------------------------------

    fn source(rx: Receiver<Message>, stop: &Arc<AtomicBool>) -> (LiveSource, Rc<RefCell<Vec<String>>>) {
        let warnings = Rc::new(RefCell::new(Vec::new()));
        let seen = Rc::clone(&warnings);
        let warn = Box::new(move |text: &str| seen.borrow_mut().push(text.to_string()));
        (LiveSource::from_channel("test input", rx, Arc::clone(stop), warn), warnings)
    }

    #[test]
    fn audio_comes_out_in_the_order_it_went_in() {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (mut live, _) = source(rx, &stop);
        tx.send(Message::Audio(vec![1.0, 2.0])).unwrap();
        tx.send(Message::Audio(vec![3.0])).unwrap();
        assert_eq!(live.description(), "test input");
        assert_eq!(live.next_chunk(), Ok(Some(vec![1.0, 2.0])));
        assert_eq!(live.next_chunk(), Ok(Some(vec![3.0])));
    }

    #[test]
    fn warnings_are_reported_and_the_audio_after_them_still_arrives() {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (mut live, warnings) = source(rx, &stop);
        tx.send(Message::Warning("overrun".into())).unwrap();
        tx.send(Message::Warning("again".into())).unwrap();
        tx.send(Message::Audio(vec![0.5])).unwrap();
        assert_eq!(live.next_chunk(), Ok(Some(vec![0.5])));
        assert_eq!(*warnings.borrow(), ["overrun", "again"]);
    }

    #[test]
    fn a_failure_ends_the_audio_with_an_error() {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (mut live, _) = source(rx, &stop);
        tx.send(Message::Audio(vec![1.0])).unwrap();
        tx.send(Message::Failed("the audio input failed: unplugged".into())).unwrap();
        assert_eq!(live.next_chunk(), Ok(Some(vec![1.0])), "audio before the failure is kept");
        assert_eq!(live.next_chunk(), Err("the audio input failed: unplugged".to_string()));
    }

    #[test]
    fn a_vanished_audio_thread_is_an_error_not_a_hang() {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (mut live, _) = source(rx, &stop);
        drop(tx);
        assert_eq!(live.next_chunk(), Err("the audio input stopped unexpectedly".to_string()));
    }

    #[test]
    fn a_stop_request_ends_a_wait_for_audio() {
        let (tx, rx) = mpsc::channel::<Message>();
        let stop = Arc::new(AtomicBool::new(false));
        let (mut live, _) = source(rx, &stop);
        let raise = Arc::clone(&stop);
        let helper = thread::spawn(move || {
            thread::sleep(Duration::from_millis(250));
            raise.store(true, Ordering::SeqCst);
        });
        let started = std::time::Instant::now();
        assert_eq!(live.next_chunk(), Ok(None), "no audio arrives, and the stop ends the wait");
        assert!(started.elapsed() < Duration::from_secs(2), "the wait must end soon after the stop");
        helper.join().unwrap();
        drop(tx);
    }

    #[test]
    fn a_stop_wins_over_audio_that_is_waiting() {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(true));
        let (mut live, _) = source(rx, &stop);
        tx.send(Message::Audio(vec![1.0])).unwrap();
        assert_eq!(live.next_chunk(), Ok(None));
    }

    // --- stream errors ---------------------------------------------------------

    #[test]
    fn hiccups_are_warnings_and_everything_else_is_fatal() {
        let xrun = cpal::Error::from(cpal::ErrorKind::Xrun);
        assert!(matches!(classify(&xrun), Message::Warning(text) if text.contains("overrun")));
        let moved = cpal::Error::from(cpal::ErrorKind::DeviceChanged);
        assert!(matches!(classify(&moved), Message::Warning(_)));
        let gone = cpal::Error::from(cpal::ErrorKind::DeviceNotAvailable);
        assert!(matches!(classify(&gone), Message::Failed(text) if text.starts_with("the audio input failed")));
        let other = cpal::Error::from(cpal::ErrorKind::BackendError);
        assert!(matches!(classify(&other), Message::Failed(_)));
    }

    #[test]
    fn only_the_formats_with_a_conversion_are_ranked() {
        assert!(format_rank(SampleFormat::F32) < format_rank(SampleFormat::I16));
        assert_eq!(format_rank(SampleFormat::F64), None);
        assert_eq!(format_rank(SampleFormat::I64), None);
    }
}
