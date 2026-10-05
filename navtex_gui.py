# NAVTEX Decoder
# Copyright (C) 2026 Brian Martlew
# SPDX-License-Identifier: GPL-3.0-or-later

"""
NAVTEX Decoder — Graphical Interface
======================================

A PyQt6 desktop front end for live decoding, as an alternative to the
command-line tool (navtex_decode.py). Both use the same decoding session
(navtex_session.py), outputs (navtex_outputs.py) and profiles
(navtex_config.py), so a profile behaves identically in either.

    python navtex_gui.py                    # finds navtex.toml (see below)
    python navtex_gui.py --config my.toml   # uses a specific file

The GUI decodes live audio only. Profiles with mode = "file" are left
out of its profile list (the command-line tool still runs them).

Where profiles are kept
------------------------
Unless --config is given, the GUI uses the first navtex.toml it finds:

  1. next to the program (the folder containing navtex_gui.py, or the
     packaged executable);
  2. in the user's settings folder: %APPDATA%\\NavtexDecoder on Windows,
     ~/.config/navtex-decoder on Linux.

If neither exists, a starter file with a single [default] profile is
created in the user's settings folder.

Relative log_dir and db_file paths in a profile are taken relative to the
folder containing the config file, so they work the same however the GUI
was launched. (The command-line tool takes them relative to the current
directory, which is normally the same folder.)

Threading
----------
Decoding runs on a worker thread (DecodeThread). Decoded text reaches
the window through GuiSink, which buffers it under a lock, and the
window collects it on a 100 ms timer. Warnings from any thread (the
decoder, the log outputs, or the audio driver's own callback thread)
arrive through a Qt signal, which Qt delivers safely on the GUI thread.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import re
import sqlite3
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

from PyQt6.QtCore import QObject, QPointF, QRectF, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (QAction, QActionGroup, QCloseEvent, QColor, QFontDatabase, QIcon, QPainter,
                         QPalette, QPixmap, QPolygonF, QStandardItemModel, QTextCursor)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from navtex_config import (
    STARTER_CONFIG,
    ConfigError,
    Profile,
    delete_profile,
    load_profile,
    load_profile_for_editing,
    read_profile_tables,
    save_profile,
)
from navtex_outputs import OutputSink, SqliteLogSink, TextLogSink
from navtex_session import DecodeSession, build_config, open_live_source
from navtex_step1_sampling_windowing import InputDevice, list_input_devices

APP_NAME = "NAVTEX Decoder"
CONFIG_NAME = "navtex.toml"
MAX_LINES = 5000             # lines kept in the text panel
UI_REFRESH_MS = 100          # how often new text and meters are collected
MAX_WARNINGS = 200           # distinct warnings kept in the warnings list
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Config file location
# ---------------------------------------------------------------------------

def program_dir() -> Path:
    """Folder containing the program: the executable when packaged (for
    example by PyInstaller), otherwise this source file."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def user_config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "NavtexDecoder"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "navtex-decoder"


def find_config(explicit: Optional[str] = None) -> Tuple[Path, bool]:
    """Returns (config path, created). See the module docstring for the
    search order. `created` is True if a starter file was just written."""
    if explicit:
        return Path(explicit).expanduser().resolve(), False
    for folder in (program_dir(), user_config_dir()):
        candidate = folder / CONFIG_NAME
        if candidate.is_file():
            return candidate, False
    target = user_config_dir() / CONFIG_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(STARTER_CONFIG, encoding="utf-8")
    return target, True


def resolve_path(value: str, base: Path) -> Path:
    """A profile path, made absolute relative to the config file's folder."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def live_profile_names(config_path: Path) -> List[str]:
    """Names of the profiles the GUI can run (mode "live", which is also
    the default when mode is omitted), in file order."""
    tables = read_profile_tables(str(config_path))
    return [name for name, table in tables.items() if table.get("mode", "live") == "live"]


def describe_device(device) -> str:
    return "default input device" if device is None else str(device)


# ---------------------------------------------------------------------------
# Plumbing between the decoding thread and the window
# ---------------------------------------------------------------------------

class GuiSink(OutputSink):
    """Collects decoded lines for the window.

    Its events are called on the decoding thread; drain() is called on
    the GUI thread. CR and LF are dropped: the window shows one line per
    block.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._done: List[Tuple[datetime.datetime, str]] = []
        self._current: Optional[List] = None    # [timestamp, text] of the line in progress

    def line_start(self, timestamp: datetime.datetime, strength: int) -> None:
        with self._lock:
            self._current = [timestamp, ""]

    def write(self, text: str) -> None:
        text = text.replace("\r", "").replace("\n", "")
        if text:
            with self._lock:
                if self._current is not None:
                    self._current[1] += text

    def line_end(self) -> None:
        with self._lock:
            if self._current is not None:
                self._done.append((self._current[0], self._current[1]))
                self._current = None

    def drain(self):
        """Returns (completed lines since the last call, line in progress
        or None). Each line is (timestamp, text)."""
        with self._lock:
            done, self._done = self._done, []
            partial = (self._current[0], self._current[1]) if self._current else None
        return done, partial


class WarningRelay(QObject):
    """Its `message` signal may be emitted from any thread; Qt delivers it
    to the window on the GUI thread."""
    message = pyqtSignal(str)

    def warn(self, text: str) -> None:
        self.message.emit(text)


class DecodeThread(QThread):
    """Runs DecodeSession.run() off the GUI thread."""
    failed = pyqtSignal(str)

    def __init__(self, session: DecodeSession, parent=None):
        super().__init__(parent)
        self.session = session

    def run(self) -> None:
        try:
            self.session.run()
        except Exception as e:  # noqa: BLE001 -- reported to the user, not swallowed
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

BAR_STYLE = """
QProgressBar {{
    border: 1px solid palette(mid);
    border-radius: 3px;
    background: palette(base);
    min-height: 16px;
    max-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {colour};
    border-radius: 2px;
}}
"""

# Meter fill colours: bright enough to stand out on a light or a dark
# background. Meter readings are shown in labels beside the bars, in the
# normal text colour, so they never sit on top of these.
COLOUR_SIGNAL = "#2a78d6"
COLOUR_GOOD = "#1a9e5f"
COLOUR_WARN = "#c98500"
COLOUR_BAD = "#d03b3b"
COLOUR_IDLE = "#8a8a8a"

# Badge (background, text) pairs, each with a contrast ratio of at least
# 4.5:1 (the WCAG level for normal text).
BADGE_GOOD = ("#157a49", "#ffffff")
BADGE_WARN = ("#e0a000", "#1a1a1a")
BADGE_BAD = ("#b8322a", "#ffffff")
BADGE_IDLE = ("#6b6a66", "#ffffff")


def badge_style(colours: Tuple[str, str], selector: str = "QLabel", padding: str = "2px 10px") -> str:
    background, text = colours
    return (f"{selector} {{ background: {background}; color: {text}; border-radius: 4px;"
            f" padding: {padding}; font-weight: bold; }}")


class Meter(QProgressBar):
    """A progress bar used as a level meter, recoloured only when its
    colour band actually changes."""

    def __init__(self, maximum: int, colour: str, parent=None):
        super().__init__(parent)
        self.setRange(0, maximum)
        self.setTextVisible(False)
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._colour = None
        self.set_colour(colour)

    def set_colour(self, colour: str) -> None:
        if colour != self._colour:
            self._colour = colour
            self.setStyleSheet(BAR_STYLE.format(colour=colour))

    def refresh_style(self) -> None:
        """Re-applies the style sheet, so palette() references in it pick
        up a new colour scheme."""
        colour, self._colour = self._colour, None
        self.set_colour(colour)


class Badge(QLabel):
    """A small coloured status label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self._state = None

    def set_state(self, text: str, colours: Tuple[str, str], tooltip: str) -> None:
        if (text, colours) == self._state:
            return
        self._state = (text, colours)
        self.setText(text)
        self.setToolTip(tooltip)
        self.setStyleSheet(badge_style(colours))


# ---------------------------------------------------------------------------
# Colour schemes
# ---------------------------------------------------------------------------
#
# "system" keeps the platform's own style and colours (on Windows 11 this
# follows the system light/dark setting). "light" and "dark" use Qt's
# cross-platform Fusion style with the high-contrast palettes below: body
# text is at least 13:1 against its background, hint text at least 6:1.

COLOUR_SCHEMES = [("system", "System Default"), ("light", "Light"), ("dark", "Dark")]

PALETTES = {
    "light": dict(window="#f0f0f0", window_text="#000000", base="#ffffff", alternate_base="#f4f4f4",
                  text="#000000", button="#e3e3e3", button_text="#000000", bright_text="#b8322a",
                  highlight="#1f5fb4", highlighted_text="#ffffff", link="#0b4fa8",
                  tooltip_base="#ffffe1", tooltip_text="#000000", light="#ffffff", midlight="#e9e9e9",
                  mid="#9a9a9a", dark="#7a7a7a", shadow="#4a4a4a", placeholder="#555555",
                  disabled="#8a8a8a"),
    "dark": dict(window="#262626", window_text="#f0f0f0", base="#141414", alternate_base="#1e1e1e",
                 text="#f0f0f0", button="#363636", button_text="#f0f0f0", bright_text="#ff7b72",
                 highlight="#2f6fc4", highlighted_text="#ffffff", link="#7cb7ff",
                 tooltip_base="#3a3a3a", tooltip_text="#f0f0f0", light="#4a4a4a", midlight="#3e3e3e",
                 mid="#5c5c5c", dark="#101010", shadow="#000000", placeholder="#b0b0b0",
                 disabled="#7a7a7a"),
}

_ROLE_KEYS = {
    "Window": "window", "WindowText": "window_text", "Base": "base",
    "AlternateBase": "alternate_base", "Text": "text", "Button": "button",
    "ButtonText": "button_text", "BrightText": "bright_text", "Highlight": "highlight",
    "HighlightedText": "highlighted_text", "Link": "link", "LinkVisited": "link",
    "ToolTipBase": "tooltip_base", "ToolTipText": "tooltip_text", "Light": "light",
    "Midlight": "midlight", "Mid": "mid", "Dark": "dark", "Shadow": "shadow",
    "PlaceholderText": "placeholder",
}

_system_style_name: Optional[str] = None


def build_palette(colours: dict) -> QPalette:
    palette = QPalette()
    for role_name, key in _ROLE_KEYS.items():
        palette.setColor(getattr(QPalette.ColorRole, role_name), QColor(colours[key]))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(colours["disabled"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(colours["mid"]))
    return palette


def apply_colour_scheme(scheme: str) -> None:
    global _system_style_name
    app = QApplication.instance()
    if _system_style_name is None:
        _system_style_name = app.style().name()
    if scheme in PALETTES:
        QApplication.setStyle("Fusion")
        QApplication.setPalette(build_palette(PALETTES[scheme]))
    else:
        QApplication.setStyle(_system_style_name)
        QApplication.setPalette(QApplication.style().standardPalette())


def make_icon(shape: str, colour: QColor, size: int = 32) -> QIcon:
    """A plain "play" triangle or "stop" square in the given colour. Qt's
    built-in media icons are drawn in a fixed dark colour, which almost
    disappears on a dark background."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    m = size * 0.2
    if shape == "play":
        painter.drawPolygon(QPolygonF([QPointF(m * 1.2, m), QPointF(size - m, size / 2),
                                       QPointF(m * 1.2, size - m)]))
    else:
        painter.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), 2, 2)
    painter.end()
    return QIcon(pixmap)


def make_hint(label: QLabel) -> QLabel:
    """Draws a label in the palette's hint (placeholder) colour, which stays
    readable in every scheme, unlike the dimmer colour of a disabled
    widget."""
    label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
    return label


SYNC_STATES = {
    "searching": ("Searching", BADGE_IDLE,
                  "No characters are being found: there is no signal, or only noise."),
    "sync": ("Sync", BADGE_WARN,
             "Characters are being found, but no message text is getting through: usually "
             "the phasing signal sent between messages, or the start of a message."),
    "data": ("Data", BADGE_GOOD,
             "Message text is being decoded. On a weak signal some characters may be "
             "wrong: the Signal bar shows how good reception is."),
    "stopped": ("Stopped", BADGE_IDLE, "The decoder is not running."),
}


# ---------------------------------------------------------------------------
# Warnings list
# ---------------------------------------------------------------------------

class WarningsDialog(QDialog):
    """Lists recent warnings. Identical messages are collapsed into one
    entry with a count, since an audio overflow can repeat many times."""

    cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warnings (times in UTC)")
        self.resize(640, 320)
        self.view = QPlainTextEdit(readOnly=True)
        self.view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        clear = QPushButton("Clear")
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        clear.clicked.connect(lambda: self.cleared.emit())
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(clear)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addLayout(buttons)

    def show_entries(self, entries) -> None:
        lines = []
        for first, last, message, count in entries:
            when = last.strftime("%Y-%m-%d %H:%M:%S")
            repeat = f"  (x{count}, first at {first.strftime('%H:%M:%S')})" if count > 1 else ""
            lines.append(f"{when}  {message}{repeat}")
        self.view.setPlainText("\n".join(lines) if lines else "No warnings.")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self, config_path: Path, created: bool):
        super().__init__()
        self.config_path = config_path
        self.settings = QSettings("NavtexDecoder", "NavtexDecoder")
        self.setWindowTitle(APP_NAME)
        self._scheme = self.settings.value("colour_scheme", "system", type=str)
        if self._scheme not in dict(COLOUR_SCHEMES):
            self._scheme = "system"
        apply_colour_scheme(self._scheme)

        self._thread: Optional[DecodeThread] = None
        self._session: Optional[DecodeSession] = None
        self._sink: Optional[GuiSink] = None
        self._failure: Optional[str] = None
        self._started_at = 0.0
        self._lines: Deque[Tuple[datetime.datetime, str]] = deque(maxlen=MAX_LINES)
        self._partial: Optional[Tuple[datetime.datetime, str]] = None
        self._warnings: List[list] = []      # [first, last, message, count]
        self._warnings_dialog: Optional[WarningsDialog] = None

        self._relay = WarningRelay()
        self._relay.message.connect(self._on_warning)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()

        self._ui_timer = QTimer(self, interval=UI_REFRESH_MS, timeout=self._refresh)
        self._clock_timer = QTimer(self, interval=1000, timeout=self._update_elapsed)

        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(900, 600)
        self.act_timestamps.setChecked(self.settings.value("show_timestamps", True, type=bool))

        self._reload_profiles(select=self.settings.value("last_profile", "", type=str))
        self._set_running(False)
        self._created = created

    def first_run_message(self) -> None:
        """Shown once, after the window appears, if a new profile file
        was just created."""
        if self._created:
            self._created = False
            QMessageBox.information(
                self, f"Welcome to {APP_NAME}",
                f"A new profile file has been created:\n{self.config_path}\n\n"
                "Open Settings to choose your audio input device and the tone "
                "frequencies your receiver produces, then press Start.")

    # --- construction ----------------------------------------------------

    def _build_actions(self) -> None:
        self.act_start = QAction("Start", self)
        self.act_start.setToolTip("Start decoding with the selected profile")
        self.act_start.triggered.connect(self.start_decoding)
        self.act_stop = QAction("Stop", self)
        self.act_stop.setToolTip("Stop decoding")
        self.act_stop.triggered.connect(self.stop_decoding)
        self.act_settings = QAction("Settings…", self)
        self.act_settings.setToolTip("Edit the selected profile")
        self.act_settings.triggered.connect(self.edit_settings)
        self.act_new = QAction("New Profile…", self)
        self.act_new.triggered.connect(self.new_profile)
        self.act_delete = QAction("Delete Profile…", self)
        self.act_delete.triggered.connect(self.remove_profile)
        self.act_timestamps = QAction("Show Timestamps (UTC)", self, checkable=True)
        self.act_timestamps.toggled.connect(self._rerender)
        self.act_clear = QAction("Clear Text", self)
        self.act_clear.triggered.connect(self.clear_text)
        self.act_warnings = QAction("Warnings…", self)
        self.act_warnings.triggered.connect(self.show_warnings)
        self.act_quit = QAction("Exit", self)
        self.act_quit.triggered.connect(self.close)
        self.act_about = QAction("About", self)
        self.act_about.triggered.connect(self.show_about)
        self._update_icons()
        self.scheme_group = QActionGroup(self)
        self.scheme_group.setExclusive(True)
        self.scheme_actions = {}
        for key, label in COLOUR_SCHEMES:
            action = QAction(label, self, checkable=True)
            action.setChecked(key == self._scheme)
            action.triggered.connect(lambda _checked, k=key: self.set_colour_scheme(k))
            self.scheme_group.addAction(action)
            self.scheme_actions[key] = action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        m = bar.addMenu("&File")
        m.addAction(self.act_quit)
        m = bar.addMenu("&Profile")
        m.addAction(self.act_settings)
        m.addSeparator()
        m.addAction(self.act_new)
        m.addAction(self.act_delete)
        m = bar.addMenu("&View")
        m.addAction(self.act_timestamps)
        m.addAction(self.act_clear)
        schemes = m.addMenu("Colour Scheme")
        for key, _label in COLOUR_SCHEMES:
            schemes.addAction(self.scheme_actions[key])
        m.addSeparator()
        m.addAction(self.act_warnings)
        m = bar.addMenu("&Help")
        m.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        tb.addWidget(QLabel(" Profile: "))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(180)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        tb.addWidget(self.profile_combo)
        tb.addSeparator()
        tb.addAction(self.act_start)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_settings)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.act_clear)

    def _build_central(self) -> None:
        self.text = QPlainTextEdit(readOnly=True)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(max(font.pointSize(), 10))
        self.text.setFont(font)
        self.text.setMaximumBlockCount(MAX_LINES + 1)    # +1: the line in progress
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text.setUndoRedoEnabled(False)

        self.signal_meter = Meter(99, COLOUR_SIGNAL)
        self.signal_meter.setToolTip("Decoding quality, from 00 (unreadable) to 99 (perfect). "
                                     "This is the same reading recorded in the logs.")
        self.audio_meter = Meter(600, COLOUR_GOOD)       # tenths of a dB above -60 dBFS
        self.audio_meter.setToolTip(
            "Peak level of the audio from the sound card, in dB below full scale.\n"
            "Keep it in the green, below about -6 dB, and never in the red (clipping).\n"
            "The decoder does not need a loud signal: -30 dB is fine.")
        self.sync_badge = Badge()
        self.signal_value = QLabel()
        self.audio_value = QLabel()
        width = self.fontMetrics().horizontalAdvance("CLIPPING") + 12
        for label, meter in ((self.signal_value, self.signal_meter), (self.audio_value, self.audio_meter)):
            label.setMinimumWidth(width)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.setToolTip(meter.toolTip())

        meters = QGridLayout()
        meters.setContentsMargins(6, 4, 6, 4)
        meters.addWidget(QLabel("Signal"), 0, 0)
        meters.addWidget(self.signal_meter, 0, 1)
        meters.addWidget(self.signal_value, 0, 2)
        meters.addWidget(QLabel("Audio"), 0, 3)
        meters.addWidget(self.audio_meter, 0, 4)
        meters.addWidget(self.audio_value, 0, 5)
        meters.addWidget(self.sync_badge, 0, 6)
        meters.setColumnStretch(1, 3)
        meters.setColumnStretch(4, 2)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.addWidget(self.text, 1)
        layout.addLayout(meters)
        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self.state_label = QLabel()
        self.device_label = QLabel()
        self.files_label = QLabel()
        self.elapsed_label = QLabel()
        self.warnings_button = QToolButton()
        self.warnings_button.setAutoRaise(True)
        self.warnings_button.clicked.connect(self.show_warnings)
        self.warnings_button.setToolTip("Show recent warnings")
        widgets = (self.state_label, self.device_label, self.files_label,
                   self.elapsed_label, self.warnings_button)
        self._separators = {}
        for i, w in enumerate(widgets):
            if i:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.VLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                sb.addPermanentWidget(line)
                self._separators[w] = line
            sb.addPermanentWidget(w)
        self._unread_warnings = False
        self._update_warnings_button()

    # --- profiles --------------------------------------------------------

    def current_profile_name(self) -> Optional[str]:
        name = self.profile_combo.currentText()
        return name or None

    def _reload_profiles(self, select: Optional[str] = None) -> None:
        select = select or self.current_profile_name()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        try:
            names = live_profile_names(self.config_path)
        except ConfigError as e:
            names = []
            self._error("Could not read profiles", str(e))
        self.profile_combo.addItems(names)
        if select in names:
            self.profile_combo.setCurrentText(select)
        self.profile_combo.blockSignals(False)
        self._on_profile_changed()

    def _on_profile_changed(self, *_args) -> None:
        name = self.current_profile_name()
        if name:
            self.settings.setValue("last_profile", name)
        self._update_info_labels()
        self._update_actions()

    def _load_selected_profile(self) -> Optional[Profile]:
        name = self.current_profile_name()
        if not name:
            return None
        try:
            return load_profile(str(self.config_path), name)
        except ConfigError as e:
            self._error("Profile problem", f"The profile [{name}] could not be used:\n\n{e}")
            return None

    def new_profile(self) -> None:
        while True:
            name, ok = QInputDialog.getText(self, "New Profile",
                                            "Name for the new profile\n"
                                            "(letters, numbers, - and _ only):")
            if not ok:
                return
            name = name.strip()
            try:
                existing = read_profile_tables(str(self.config_path))
            except ConfigError as e:
                self._error("Could not read profiles", str(e))
                return
            if not PROFILE_NAME_RE.match(name):
                QMessageBox.warning(self, "New Profile",
                                    "Please use only letters, numbers, - and _ in the name.")
            elif name in existing:
                QMessageBox.warning(self, "New Profile", f"There is already a profile called {name!r}.")
            else:
                break
        # Start from a copy of the selected profile, so the device and tone
        # settings carry over; otherwise from the built-in defaults.
        base = None
        if self.current_profile_name():
            try:
                base = load_profile(str(self.config_path), self.current_profile_name())
            except ConfigError:
                base = None
        profile = dataclasses.replace(base or Profile(), mode="live", wav_file=None)
        try:
            save_profile(str(self.config_path), name, profile)
        except (ConfigError, OSError) as e:
            self._error("Could not create the profile", str(e))
            return
        self._reload_profiles(select=name)
        self.edit_settings()

    def remove_profile(self) -> None:
        name = self.current_profile_name()
        if not name:
            return
        answer = QMessageBox.question(
            self, "Delete Profile",
            f"Delete the profile [{name}]?\n\nIts log files and database are not affected.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_profile(str(self.config_path), name)
        except (ConfigError, OSError) as e:
            self._error("Could not delete the profile", str(e))
            return
        self._reload_profiles()

    def edit_settings(self) -> None:
        name = self.current_profile_name()
        if not name:
            return
        try:
            profile, problem = load_profile_for_editing(str(self.config_path), name)
        except ConfigError as e:
            self._error("Profile problem", str(e))
            return
        dialog = SettingsDialog(self, self.config_path, name, profile, problem)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._reload_profiles(select=name)

    # --- decoding --------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None

    def start_decoding(self) -> None:
        if self.is_running():
            return
        profile = self._load_selected_profile()
        if profile is None:
            return
        base = self.config_path.parent
        warn = self._relay.warn
        try:
            source, description = open_live_source(build_config(profile), profile.device, warn=warn)
        except Exception as e:  # noqa: BLE001 -- PortAudio raises several types
            self._error("Could not open the audio device",
                        f"The input device ({describe_device(profile.device)}) could not be "
                        f"opened at {profile.sample_rate} Hz.\n\n{e}\n\n"
                        "Check the device is connected, or choose another in Settings.")
            return

        self._sink = GuiSink()
        sinks: List[OutputSink] = [self._sink]
        try:
            if profile.log_dir:
                sinks.append(TextLogSink(str(resolve_path(profile.log_dir, base)), description, warn=warn))
            if profile.db_file:
                sinks.append(SqliteLogSink(str(resolve_path(profile.db_file, base)), warn=warn))
        except (OSError, sqlite3.Error) as e:
            for sink in sinks:
                sink.close()
            source.close()
            self._sink = None
            self._error("Could not open the log output", str(e))
            return

        self._session = DecodeSession(source, profile, sinks, warn=warn)
        self._failure = None
        self._thread = DecodeThread(self._session, self)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(self._on_finished)
        self._started_at = time.monotonic()
        self._thread.start()
        self._set_running(True)
        self._update_info_labels(profile, sinks)

    def stop_decoding(self) -> None:
        if self._session is not None:
            self.state_label.setText("Stopping…")
            self.act_stop.setEnabled(False)
            self._session.stop()

    def _on_failed(self, message: str) -> None:
        self._failure = message

    def _on_finished(self) -> None:
        self._refresh()                 # collect anything still buffered
        thread, self._thread = self._thread, None
        self._session = None
        self._sink = None
        if thread is not None:
            thread.deleteLater()
        self._set_running(False)
        if self._failure:
            self._error("Decoding stopped", f"Decoding stopped because of an error:\n\n{self._failure}")
            self._failure = None

    # --- periodic updates ------------------------------------------------

    def _refresh(self) -> None:
        if self._sink is not None:
            done, partial = self._sink.drain()
            if done or partial != self._partial:
                self._lines.extend(done)
                self._partial = partial
                self._append(done, partial)
        session = self._session
        if session is not None:
            self.signal_meter.setValue(session.tracker.level)
            self.signal_value.setText(f"{session.tracker.level:02d}")
            self._show_audio_level(session.audio_level.peak_dbfs, session.audio_level.clipping)
            self.sync_badge.set_state(*SYNC_STATES[session.sync_state])

    def _show_audio_level(self, dbfs: float, clipping: bool) -> None:
        self.audio_meter.setValue(int(round(max(0.0, min(60.0, dbfs + 60.0)) * 10)))
        if clipping:
            self.audio_meter.set_colour(COLOUR_BAD)
            self.audio_value.setText("CLIPPING")
        elif dbfs <= -90.0:
            self.audio_meter.set_colour(COLOUR_IDLE)
            self.audio_value.setText("No audio")
        else:
            self.audio_meter.set_colour(COLOUR_WARN if dbfs > -6.0 else COLOUR_GOOD)
            self.audio_value.setText(f"{dbfs:.0f} dB")

    def _update_elapsed(self) -> None:
        if self.is_running():
            s = int(time.monotonic() - self._started_at)
            self.elapsed_label.setText(f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}")

    # --- text panel ------------------------------------------------------

    def _format(self, line: Tuple[datetime.datetime, str]) -> str:
        timestamp, text = line
        return f"{timestamp.strftime('%H:%M:%S')}  {text}" if self.act_timestamps.isChecked() else text

    def _append(self, done, partial) -> None:
        # The document's last block always holds the line in progress (or
        # is empty); completed lines are written into it and a new empty
        # block is started after each.
        bar = self.text.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2
        cursor = QTextCursor(self.text.document())
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        for line in done:
            cursor.insertText(self._format(line))
            cursor.insertBlock()
        if partial is not None:
            cursor.insertText(self._format(partial))
        cursor.endEditBlock()
        if at_bottom:
            bar.setValue(bar.maximum())

    def _rerender(self, *_args) -> None:
        self.settings.setValue("show_timestamps", self.act_timestamps.isChecked())
        text = "".join(self._format(line) + "\n" for line in self._lines)
        if self._partial is not None:
            text += self._format(self._partial)
        self.text.setPlainText(text)
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def clear_text(self) -> None:
        self._lines.clear()
        self._partial = None
        self.text.clear()

    # --- warnings --------------------------------------------------------

    def _on_warning(self, message: str) -> None:
        message = " ".join(message.split())
        now = datetime.datetime.now(datetime.timezone.utc)
        for entry in self._warnings:
            if entry[2] == message:
                entry[1] = now
                entry[3] += 1
                self._warnings.remove(entry)
                self._warnings.append(entry)
                break
        else:
            self._warnings.append([now, now, message, 1])
            del self._warnings[:-MAX_WARNINGS]
        self._unread_warnings = not (self._warnings_dialog is not None
                                     and self._warnings_dialog.isVisible())
        self._update_warnings_button()
        if self._warnings_dialog is not None and self._warnings_dialog.isVisible():
            self._warnings_dialog.show_entries(self._warnings)

    def _update_warnings_button(self) -> None:
        n = sum(entry[3] for entry in self._warnings)
        self._show_in_status_bar(self.warnings_button, n > 0)
        self.warnings_button.setText(f"⚠ {n} warning{'s' if n != 1 else ''}")
        self.warnings_button.setStyleSheet(
            badge_style(BADGE_WARN, "QToolButton", "0 6px") if self._unread_warnings else "")

    def show_warnings(self) -> None:
        if self._warnings_dialog is None:
            self._warnings_dialog = WarningsDialog(self)
            self._warnings_dialog.cleared.connect(self._clear_warnings)
        self._warnings_dialog.show_entries(self._warnings)
        self._warnings_dialog.show()
        self._warnings_dialog.raise_()
        self._unread_warnings = False
        self._update_warnings_button()

    def _clear_warnings(self) -> None:
        self._warnings.clear()
        self._unread_warnings = False
        self._update_warnings_button()
        if self._warnings_dialog is not None:
            self._warnings_dialog.show_entries(self._warnings)

    # --- state display ---------------------------------------------------

    def _set_running(self, running: bool) -> None:
        if running:
            self._ui_timer.start()
            self._clock_timer.start()
            self.state_label.setText("Listening")
            self.elapsed_label.setText("00:00:00")
            self._show_in_status_bar(self.elapsed_label, True)
        else:
            self._ui_timer.stop()
            self._clock_timer.stop()
            self.state_label.setText("Stopped")
            self._show_in_status_bar(self.elapsed_label, False)
            self.signal_meter.setValue(0)
            self.signal_value.setText("--")
            self.audio_meter.setValue(0)
            self.audio_meter.set_colour(COLOUR_IDLE)
            self.audio_value.setText("--")
            self.sync_badge.set_state(*SYNC_STATES["stopped"])
            self._update_info_labels()
        self._update_actions()

    def _show_in_status_bar(self, widget: QWidget, visible: bool) -> None:
        widget.setVisible(visible)
        if widget in self._separators:
            self._separators[widget].setVisible(visible)

    def _update_actions(self) -> None:
        running = self.is_running()
        has_profile = self.current_profile_name() is not None
        self.act_start.setEnabled(not running and has_profile)
        self.act_stop.setEnabled(running)
        self.act_settings.setEnabled(not running and has_profile)
        self.act_new.setEnabled(not running)
        self.act_delete.setEnabled(not running and has_profile)
        self.profile_combo.setEnabled(not running)

    def _update_info_labels(self, profile: Optional[Profile] = None, sinks=None) -> None:
        if not self.current_profile_name():
            self.device_label.setText("No profile: use Profile > New Profile")
            self.files_label.setText("")
            self.setWindowTitle(APP_NAME)
            return
        self.setWindowTitle(f"{APP_NAME} — {self.current_profile_name()}")
        if profile is None:
            try:
                profile = load_profile(str(self.config_path), self.current_profile_name())
            except ConfigError:
                self.device_label.setText("Profile has an error: open Settings to fix it")
                self.files_label.setText("")
                return
        self.device_label.setText(f"Device: {describe_device(profile.device)}")
        files = []
        for sink in sinks or []:
            if isinstance(sink, TextLogSink):
                files.append(f"Log: {sink.path.name}")
            elif isinstance(sink, SqliteLogSink):
                files.append(f"Database: {sink.path.name}")
        if sinks is None:
            if profile.log_dir:
                files.append("Log: on")
            if profile.db_file:
                files.append(f"Database: {Path(profile.db_file).name}")
        self.files_label.setText("   ".join(files) if files else "Not logging")
        tips = []
        base = self.config_path.parent
        if profile.log_dir:
            tips.append(f"Log folder: {resolve_path(profile.log_dir, base)}")
        if profile.db_file:
            tips.append(f"Database: {resolve_path(profile.db_file, base)}")
        self.files_label.setToolTip("\n".join(tips))

    # --- colour scheme ---------------------------------------------------

    def _update_icons(self) -> None:
        colour = QApplication.palette().color(QPalette.ColorRole.ButtonText)
        self.act_start.setIcon(make_icon("play", colour))
        self.act_stop.setIcon(make_icon("stop", colour))

    def set_colour_scheme(self, scheme: str) -> None:
        self._scheme = scheme
        self.settings.setValue("colour_scheme", scheme)
        apply_colour_scheme(scheme)
        self.scheme_actions[scheme].setChecked(True)
        # Style sheets that refer to palette() colours are only resolved when
        # set, so re-apply them for the new palette.
        self.signal_meter.refresh_style()
        self.audio_meter.refresh_style()
        self._update_warnings_button()
        self._update_icons()

    # --- misc ------------------------------------------------------------

    def _error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def show_about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><p>Decodes NAVTEX maritime safety broadcasts "
            "(100-baud FSK, CCIR 476 / SITOR-B) from a receiver's audio output.</p>"
            f"<p>Profiles: {self.config_path}</p>")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_running():
            self._session.stop()
            self._thread.wait(5000)
            self._on_finished()
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

# Advanced settings: (field, label, kind, minimum, maximum, step, decimals, tooltip)
ADVANCED_FIELDS = [
    ("sample_rate", "Sample rate (Hz)", int, 8000, 192000, 1000, 0,
     "Audio sample rate. The input device must support it."),
    ("oversample", "Frames per bit", int, 2, 32, 1, 0,
     "Analysis frames per bit. Affects every later stage; best left at 8."),
    ("window_type", "Window function", str, 0, 0, 0, 0,
     "Window applied to each frame. The signal-strength scale is calibrated for hamming."),
    ("loop_gain", "Bit-clock loop gain", float, 0.001, 1.0, 0.01, 3,
     "How strongly the bit clock corrects timing. Lower is steadier on weak signals; "
     "higher locks faster on strong ones."),
    ("sync_window", "Character sync window (bits)", int, 14, 5000, 10, 0,
     "Recent bits used to find character alignment. Must exceed 7 x minimum groups."),
    ("min_groups_for_acquire", "Minimum groups to acquire", int, 1, 500, 1, 0,
     "Characters needed before alignment can lock. Lower is faster but less certain."),
    ("char_acquire_threshold", "Character acquire threshold", float, 0.0, 1.0, 0.01, 2,
     "Fraction of valid characters needed to lock alignment (noise gives about 0.27)."),
    ("char_drop_threshold", "Character drop threshold", float, 0.0, 1.0, 0.01, 2,
     "Below this, the decoder looks for a better alignment."),
    ("char_switch_margin", "Character switch margin", float, 0.0, 1.0, 0.01, 2,
     "How much better another alignment must be before switching to it."),
    ("fec_acquire_threshold", "FEC acquire threshold", float, 0.0, 1.0, 0.01, 2,
     "Match rate needed to lock the repeated-character interleave."),
    ("fec_switch_margin", "FEC switch margin", float, 0.0, 1.0, 0.01, 2,
     "How much better the other interleave must be before switching."),
    ("min_samples_for_rate", "FEC minimum comparisons", int, 1, 200, 1, 0,
     "Comparisons needed before a match rate is trusted. Must not exceed the rate window."),
    ("rate_window", "FEC rate window", int, 1, 500, 1, 0,
     "Recent comparisons used for each match rate."),
    ("lock_window", "FEC lock window", int, 1, 500, 1, 0,
     "With the rate window, sets how much history is replayed when FEC locks."),
    ("phasing_burst_threshold", "Phasing burst length", int, 1, 100, 1, 0,
     "Consecutive phasing characters treated as a burst, after which FEC re-locks."),
    ("signal_strength_window", "Signal strength averaging (bits)", int, 10, 5000, 10, 0,
     "Bits averaged for the signal-strength reading (100 bits is 1 second)."),
]

WINDOW_TYPES = ["hamming", "hann", "blackman", "blackmanharris", "nuttall", "bartlett", "boxcar"]


class SettingsDialog(QDialog):
    """Edits one profile. Nothing is saved until OK, and OK only closes
    the dialog once the profile has passed validation and been written."""

    def __init__(self, parent, config_path: Path, name: str, profile: Profile,
                 problem: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Settings — {name}")
        self.config_path = config_path
        self.name = name
        self.profile = profile
        self.defaults = Profile()
        self._devices: List[InputDevice] = []

        tabs = QTabWidget()
        tabs.addTab(self._build_basic_tab(), "Basic")
        tabs.addTab(self._build_advanced_tab(), "Advanced")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        if problem:
            note = QLabel(f"This profile needs fixing before it can be used:\n{problem}")
            note.setWordWrap(True)
            note.setStyleSheet(badge_style(BADGE_BAD, padding="6px").replace(" font-weight: bold;", ""))
            layout.addWidget(note)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        self.resize(640, 600)
        self._load_values(profile)

    # --- basic tab -------------------------------------------------------

    def _build_basic_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(30)
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Look for audio devices again (for example after plugging one in)")
        refresh.clicked.connect(lambda: self._fill_devices(self._selected_device()))
        row = QHBoxLayout()
        row.addWidget(self.device_combo, 1)
        row.addWidget(refresh)
        form.addRow("Input device:", row)
        self.device_note = QLabel()
        self.device_note.setWordWrap(True)
        self.device_note.setStyleSheet(badge_style(BADGE_BAD, padding="4px").replace(" font-weight: bold;", ""))
        self.device_note.hide()
        form.addRow("", self.device_note)

        self.mark_spin = self._freq_spin()
        self.space_spin = self._freq_spin()
        tip = ("The audio frequencies of the two NAVTEX tones as your receiver produces them. "
               "They depend on the receiver's tuning and mode, not on the NAVTEX standard. "
               "If nothing decodes, try Swap.")
        self.mark_spin.setToolTip(tip)
        self.space_spin.setToolTip(tip)
        swap = QPushButton("Swap")
        swap.setToolTip("Swap the mark and space frequencies (if your receiver inverts them)")
        swap.clicked.connect(self._swap_tones)
        tones = QHBoxLayout()
        tones.addWidget(QLabel("Mark"))
        tones.addWidget(self.mark_spin)
        tones.addWidget(QLabel("Space"))
        tones.addWidget(self.space_spin)
        tones.addWidget(swap)
        tones.addStretch()
        form.addRow("Tone frequencies:", tones)
        self.tone_note = QLabel()
        form.addRow("", self.tone_note)
        self.mark_spin.valueChanged.connect(self._update_tone_note)
        self.space_spin.valueChanged.connect(self._update_tone_note)

        self.log_check = QCheckBox("Save decoded text to log files in this folder:")
        self.log_edit = QLineEdit()
        self.log_browse = log_browse = QPushButton("Browse…")
        log_browse.clicked.connect(self._browse_log_dir)
        form.addRow(self.log_check)
        row = QHBoxLayout()
        row.addWidget(self.log_edit, 1)
        row.addWidget(log_browse)
        form.addRow("", row)
        self.log_check.toggled.connect(self.log_edit.setEnabled)
        self.log_check.toggled.connect(log_browse.setEnabled)

        self.db_check = QCheckBox("Save decoded lines to this database file:")
        self.db_edit = QLineEdit()
        self.db_browse = db_browse = QPushButton("Browse…")
        db_browse.clicked.connect(self._browse_db_file)
        form.addRow(self.db_check)
        row = QHBoxLayout()
        row.addWidget(self.db_edit, 1)
        row.addWidget(db_browse)
        form.addRow("", row)
        hint = QLabel("Use a file on a local disk, not a network or cloud-synced drive. "
                      "An existing database is added to, not replaced.")
        hint.setWordWrap(True)
        make_hint(hint)
        form.addRow("", hint)
        self.db_check.toggled.connect(self.db_edit.setEnabled)
        self.db_check.toggled.connect(db_browse.setEnabled)

        note = QLabel(f"Relative paths are relative to {self.config_path.parent}")
        note.setWordWrap(True)
        make_hint(note)
        form.addRow("", note)
        return page

    def _freq_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(100.0, 5000.0)
        spin.setDecimals(1)
        spin.setSingleStep(5.0)
        spin.setSuffix(" Hz")
        return spin

    def _swap_tones(self) -> None:
        mark, space = self.mark_spin.value(), self.space_spin.value()
        self.mark_spin.setValue(space)
        self.space_spin.setValue(mark)

    def _update_tone_note(self) -> None:
        mark, space = self.mark_spin.value(), self.space_spin.value()
        shift = abs(mark - space)
        text = f"Centre {(mark + space) / 2:.0f} Hz, shift {shift:.0f} Hz"
        if abs(shift - 170) > 10:
            text += "  (NAVTEX uses a 170 Hz shift)"
        self.tone_note.setText(text)

    def _browse_log_dir(self) -> None:
        start = self.log_edit.text() or str(self.config_path.parent)
        folder = QFileDialog.getExistingDirectory(self, "Log folder", start)
        if folder:
            self.log_edit.setText(folder)

    def _browse_db_file(self) -> None:
        start = self.db_edit.text() or str(self.config_path.parent / f"{self.name}.db")
        path, _ = QFileDialog.getSaveFileName(
            self, "Database file", start, "SQLite database (*.db *.sqlite);;All files (*)",
            options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.db_edit.setText(path)

    # --- devices ---------------------------------------------------------

    def _fill_devices(self, wanted) -> None:
        combo = self.device_combo
        combo.clear()
        combo.addItem("System default input device", None)
        self.device_note.hide()
        try:
            self._devices = list_input_devices(self._int_value("sample_rate"))
        except Exception as e:  # noqa: BLE001 -- sounddevice/PortAudio problems
            self._devices = []
            self.device_note.setText(f"Could not list audio devices: {e}")
            self.device_note.show()
        model = combo.model()
        for dev in self._devices:
            label = f"{dev.name}  ({dev.hostapi})"
            if dev.is_default:
                label += "  [default]"
            if not dev.usable:
                label += f"  - cannot capture at {self._int_value('sample_rate')} Hz"
            combo.addItem(label, dev.query)
            if not dev.usable and isinstance(model, QStandardItemModel):
                model.item(combo.count() - 1).setEnabled(False)

        index = self._find_device(wanted)
        if index is None:
            combo.addItem(f"{wanted}  (saved setting, not currently found)", wanted)
            index = combo.count() - 1
        combo.setCurrentIndex(index)

    def _find_device(self, wanted) -> Optional[int]:
        if wanted is None:
            return 0
        for i, dev in enumerate(self._devices, start=1):
            if isinstance(wanted, int) and dev.index == wanted:
                return i
            if isinstance(wanted, str) and wanted.lower() in (dev.query.lower(), dev.name.lower()):
                return i
        return None

    def _selected_device(self):
        return self.device_combo.currentData()

    # --- advanced tab ----------------------------------------------------

    def _build_advanced_tab(self) -> QWidget:
        inner = QWidget()
        grid = QGridLayout(inner)
        intro = QLabel("These settings tune the decoder itself. Most users should leave "
                       "them at their defaults.")
        intro.setWordWrap(True)
        grid.addWidget(intro, 0, 0, 1, 3)
        self.fields = {}
        for row, (name, label, kind, lo, hi, step, decimals, tip) in enumerate(ADVANCED_FIELDS, start=1):
            if kind is str:
                widget = QComboBox()
                widget.setEditable(True)
                widget.addItems(WINDOW_TYPES)
            elif kind is int:
                widget = QSpinBox()
                widget.setRange(int(lo), int(hi))
                widget.setSingleStep(int(step))
            else:
                widget = QDoubleSpinBox()
                widget.setRange(lo, hi)
                widget.setDecimals(decimals)
                widget.setSingleStep(step)
            widget.setToolTip(tip)
            name_label = QLabel(label)
            name_label.setToolTip(tip)
            default = make_hint(QLabel(f"default {getattr(self.defaults, name)}"))
            grid.addWidget(name_label, row, 0)
            grid.addWidget(widget, row, 1)
            grid.addWidget(default, row, 2)
            self.fields[name] = widget
        reset = QPushButton("Reset All to Defaults")
        reset.clicked.connect(self._reset_advanced)
        grid.addWidget(reset, len(ADVANCED_FIELDS) + 1, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)
        grid.setRowStretch(len(ADVANCED_FIELDS) + 2, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _set_field(self, name: str, value) -> None:
        widget = self.fields[name]
        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        else:
            widget.setValue(value)

    def _int_value(self, name: str) -> int:
        return int(self.fields[name].value())

    def _reset_advanced(self) -> None:
        for name, *_ in ADVANCED_FIELDS:
            self._set_field(name, getattr(self.defaults, name))

    # --- load / save -----------------------------------------------------

    def _load_values(self, p: Profile) -> None:
        for name, *_ in ADVANCED_FIELDS:
            self._set_field(name, getattr(p, name))
        self.mark_spin.setValue(p.mark_freq)
        self.space_spin.setValue(p.space_freq)
        self._update_tone_note()
        self.log_check.setChecked(bool(p.log_dir))
        self.log_edit.setText(p.log_dir or "")
        self.log_edit.setEnabled(bool(p.log_dir))
        self.log_browse.setEnabled(bool(p.log_dir))
        self.db_check.setChecked(bool(p.db_file))
        self.db_edit.setText(p.db_file or "")
        self.db_edit.setEnabled(bool(p.db_file))
        self.db_browse.setEnabled(bool(p.db_file))
        self._fill_devices(p.device)

    def _collect(self) -> Profile:
        values = {}
        for name, _label, kind, *_ in ADVANCED_FIELDS:
            widget = self.fields[name]
            if kind is str:
                values[name] = widget.currentText().strip()
            elif kind is int:
                values[name] = int(widget.value())
            else:
                values[name] = round(float(widget.value()), 6)
        log_dir = self.log_edit.text().strip() if self.log_check.isChecked() else ""
        db_file = self.db_edit.text().strip() if self.db_check.isChecked() else ""
        return dataclasses.replace(
            self.profile, mode="live", wav_file=None,
            device=self._selected_device(),
            mark_freq=round(self.mark_spin.value(), 1),
            space_freq=round(self.space_spin.value(), 1),
            log_dir=log_dir or None, db_file=db_file or None,
            **values)

    def _save(self) -> None:
        profile = self._collect()
        problems = []
        if self.log_check.isChecked() and not profile.log_dir:
            problems.append("Choose a folder for the log files, or untick that option.")
        if self.db_check.isChecked() and not profile.db_file:
            problems.append("Choose a database file, or untick that option.")
        try:
            from scipy.signal import get_window
            get_window(profile.window_type, 16)
        except Exception:  # noqa: BLE001 -- scipy raises ValueError or others
            problems.append(f"{profile.window_type!r} is not a window function scipy recognises.")
        if problems:
            QMessageBox.warning(self, "Settings", "\n\n".join(problems))
            return
        try:
            save_profile(str(self.config_path), self.name, profile)
        except ConfigError as e:
            QMessageBox.warning(self, "Settings", f"These settings can't be used:\n\n{e}")
            return
        except OSError as e:
            QMessageBox.critical(self, "Settings", f"Could not save the profile:\n\n{e}")
            return
        self.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="NAVTEX Decoder graphical interface")
    parser.add_argument("--config", help=f"Profile file to use (default: find {CONFIG_NAME})")
    args, qt_args = parser.parse_known_args(argv)

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    try:
        config_path, created = find_config(args.config)
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
    except (ConfigError, OSError) as e:
        QMessageBox.critical(None, APP_NAME, f"Could not open the profile file:\n\n{e}")
        return 1
    window = MainWindow(config_path, created)
    window.show()
    QTimer.singleShot(0, window.first_run_message)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
