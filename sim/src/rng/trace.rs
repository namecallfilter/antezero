//! Per-action diagnostic timeline (`trace-keys` builds) and its no-op
//! stand-in when the feature is off.

#[cfg(feature = "trace-keys")]
use std::fmt;

/// Records a timeline entry in a [`PseudoRandom`](super::PseudoRandom)'s trace
/// (`trace-keys` builds only).
macro_rules! trace {
	($rng:expr, $($t:tt)*) => {
		#[cfg(feature = "trace-keys")]
		$rng.trace.record(format_args!($($t)*));
	};
}
pub(crate) use trace;

/// The diagnostic timeline of the current action (feature `trace-keys`,
/// used by the oracle differential tests): `"<frame> <what>"` entries for
/// every `pseudoseed` call, card draw and card removal, and optionally a log
/// of completed events. It lives with the RNG because every random draw is a
/// timeline entry.
#[cfg(feature = "trace-keys")]
#[derive(Clone, Debug)]
pub(crate) struct Trace {
	/// Whether entries are recorded (on by default in feature builds).
	enabled: bool,
	/// Frame of the current action, stamped on every entry.
	frame: u32,
	entries: Vec<String>,
	/// Completed-event log, when enabled.
	event_log: Option<Vec<String>>,
}

#[cfg(feature = "trace-keys")]
impl Default for Trace {
	fn default() -> Self {
		Trace {
			enabled: true,
			frame: 0,
			entries: Vec::new(),
			event_log: None,
		}
	}
}

#[cfg(feature = "trace-keys")]
impl Trace {
	#[inline]
	pub(crate) fn record(&mut self, what: fmt::Arguments<'_>) {
		if self.enabled {
			self.entries.push(format!("{} {what}", self.frame));
		}
	}

	/// Stamps the entries that follow with `frame`.
	#[inline]
	pub(crate) fn set_frame(&mut self, frame: u32) {
		self.frame = frame;
	}

	/// Starts the timeline of a new action.
	pub(crate) fn clear(&mut self) {
		self.frame = 0;
		self.entries.clear();
		if let Some(l) = self.event_log.as_mut() {
			l.clear();
		}
	}

	/// Turns recording on or off; off also drops the timeline.
	pub(crate) fn set_enabled(&mut self, on: bool) {
		self.enabled = on;
		if !on {
			self.entries = Vec::new();
		}
	}

	pub(crate) fn entries(&self) -> &[String] {
		&self.entries
	}

	/// Starts logging the completed events of each action.
	pub(crate) fn enable_event_log(&mut self) {
		self.event_log = Some(Vec::new());
	}

	/// The completed events of the current action, if logging is on.
	pub(crate) fn event_log(&self) -> &[String] {
		self.event_log.as_deref().unwrap_or(&[])
	}

	/// Logs a completed event, if logging is on.
	pub(crate) fn log_event(&mut self, what: fmt::Arguments<'_>) {
		if let Some(log) = self.event_log.as_mut() {
			log.push(what.to_string());
		}
	}
}

/// Without `trace-keys` there is no timeline: an empty stand-in whose
/// methods do nothing.
#[cfg(not(feature = "trace-keys"))]
#[derive(Clone, Debug, Default)]
pub(crate) struct Trace {}

#[cfg(not(feature = "trace-keys"))]
#[expect(
	clippy::unused_self,
	reason = "the stand-in records nothing, so its methods have nothing in `self` to use"
)]
impl Trace {
	#[inline]
	pub(crate) fn set_frame(&mut self, _frame: u32) {}

	#[inline]
	pub(crate) fn clear(&mut self) {}

	pub(crate) fn set_enabled(&mut self, _on: bool) {}

	pub(crate) fn entries(&self) -> &[String] {
		&[]
	}
}
