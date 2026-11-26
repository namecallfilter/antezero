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
//! optional destination slot (for reordering).

mod candidates;
pub mod center;
mod digest;
mod encode;
mod layout;
mod op;
mod order;
mod preview;
mod preview_items;
mod tokens;
mod zone;

pub use candidates::{
	Candidate, PREVIEW_BUDGET, PREVIEW_PLAYS, PREVIEW_REFINE, PREVIEW_SAMPLES, PlayPreview,
	Scenarios, preview_score, top_plays,
};
pub use encode::{
	CAPACITY_FLOATS, GLOB_HAS_PREVIEW, GLOB_PREVIEW, encode, encode_opts, encode_with,
	hidden_scoring_state,
};
pub use layout::{
	GLOB_FLOAT, GLOB_FLOAT_V2, GLOB_INT, GLOB_VOCAB, Layout, MAX_CANDIDATES, MAX_HAND, MAX_OPS,
	MAX_TOKENS, OP_FLOAT, OP_INT, OP_ORDER_GAIN, ObsBuf, ObsMut, PREVIEW_TOK, TOK_FLOAT, TOK_INT,
	TOK_PREVIEW, TOK_VOCAB, TokInts,
};
pub use op::{CardArg, MAX_REARRANGES, Op, OpKind, OpOptions, decode, fallback, ops};
pub use order::{ORDER_GAIN_SCALE, ORDER_JOKERS, order_gain, order_label};
pub use preview::{
	Capacity, PREVIEW_HANDS, PREVIEW_LOG_MAX, PREVIEW_RATIO_MAX, best_play, capacity, virtual_round,
};
pub use preview_items::{
	BuildPreview, ItemPreview, PREVIEW_CACHE, PREVIEW_ITEM_HANDS, build_preview,
};
pub use tokens::hand_id;
pub(crate) use tokens::unseen_pool;
pub use zone::{ItemZone, Zone};
