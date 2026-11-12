//! Fixed-size numeric observation and factored action space for learning
//! agents.
//!
//! **Observation**: a set of entity tokens (hand, jokers, consumables, shop,
//! pack, tags, blinds, poker-hand levels, remaining deck and discard pile)
//! plus a global feature vector. Only information visible to a player is
//! encoded: the deck is a sorted multiset (draw order is hidden), face-down
//! cards hide their identity, and no RNG state is exposed.
//!
//! **Actions** are factored into an *op* (what to do, optionally on a
//! target token), an optional set of hand cards (chosen by a pointer in
//! increasing index order, so each set has exactly one encoding) and an
//! optional destination slot (for reordering). [`ops`] lists the legal ops
//! with their card constraints; [`decode`] turns a choice back into an
//! [`Action`](crate::Action) and checks it.
mod candidates;
pub mod center;
mod digest;
mod encode;
mod layout;
pub mod op;
mod ops;
mod preview;
pub mod zone;

pub use candidates::{
	Candidate, PREVIEW_BUDGET, PREVIEW_REFINE, PREVIEW_SAMPLES, PlayPreview, preview_score,
	top_plays,
};
pub(crate) use digest::canon_key;
pub use encode::{encode, encode_opts, encode_with, hidden_scoring_state};
pub use layout::{
	GLOB_FLOAT, GLOB_INT, GLOB_VOCAB, Layout, MAX_CANDIDATES, MAX_HAND, MAX_OPS, MAX_TOKENS,
	OP_FLOAT, OP_INT, ObsMut, TOK_FLOAT, TOK_INT, TOK_PREVIEW, TOK_VOCAB,
};
pub use ops::{MAX_REARRANGES, Op, decode, fallback, ops};
pub use preview::{
	BuildPreview, Capacity, PREVIEW_CACHE, PREVIEW_HANDS, PREVIEW_ITEM_HANDS, PREVIEW_PLAYS,
	PREVIEW_TOK, best_play, build_preview, capacity, virtual_round,
};
