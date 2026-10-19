//! One state's encoded observation and legal ops, written into
//! caller-provided buffers: numpy rows (`VecEnv`) or owned vectors
//! (`Game.encode`, the final states of truncated episodes).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sim::game::Game;
use sim::obs::{self, ObsMut, Op};

/// Mutable views of one state's observation (row-major; see `sim::obs`).
pub(crate) struct Rows<'a> {
    pub ti: &'a mut [i32],
    pub tf: &'a mut [f32],
    pub gi: &'a mut [i32],
    pub gf: &'a mut [f32],
    pub oa: &'a mut [i32],
    pub of: &'a mut [f32],
    pub n_ops: &'a mut i32,
    pub n_tokens: &'a mut i32,
    pub n_hand: &'a mut i32,
}

impl Rows<'_> {
    /// Encodes `g` (after refreshing the joker values the game recomputes
    /// every frame) and its legal ops; `ops` receives the ops, for decoding
    /// the policy's choice.
    pub fn write(&mut self, g: &mut Game, candidates: usize, deck_tokens: bool, ops: &mut Vec<Op>) {
        g.update_derived();
        let lay = obs::encode_opts(g, &mut ObsMut { tok_i: self.ti, tok_f: self.tf, glob_i: self.gi, glob_f: self.gf }, deck_tokens);
        obs::ops(g, &lay, candidates, ops);
        self.oa.fill(0);
        self.of.fill(0.0);
        for (j, op) in ops.iter().enumerate() {
            self.oa[j * obs::OP_INT..(j + 1) * obs::OP_INT].copy_from_slice(&op.ints());
            self.of[j * obs::OP_FLOAT..(j + 1) * obs::OP_FLOAT].copy_from_slice(&op.feats);
        }
        *self.n_ops = ops.len() as i32;
        *self.n_tokens = lay.n_tokens as i32;
        *self.n_hand = lay.hand as i32;
    }

    pub fn copy_from(&mut self, r: &ObsRow) {
        self.ti.copy_from_slice(&r.ti);
        self.tf.copy_from_slice(&r.tf);
        self.gi.copy_from_slice(&r.gi);
        self.gf.copy_from_slice(&r.gf);
        self.oa.copy_from_slice(&r.oa);
        self.of.copy_from_slice(&r.of);
        *self.n_ops = r.n_ops;
        *self.n_tokens = r.n_tokens;
        *self.n_hand = r.n_hand;
    }
}

/// An owned observation.
pub(crate) struct ObsRow {
    pub ti: Vec<i32>,
    pub tf: Vec<f32>,
    pub gi: Vec<i32>,
    pub gf: Vec<f32>,
    pub oa: Vec<i32>,
    pub of: Vec<f32>,
    pub n_ops: i32,
    pub n_tokens: i32,
    pub n_hand: i32,
}

impl ObsRow {
    pub fn new() -> ObsRow {
        ObsRow {
            ti: vec![0; obs::MAX_TOKENS * obs::TOK_INT],
            tf: vec![0.0; obs::MAX_TOKENS * obs::TOK_FLOAT],
            gi: vec![0; obs::GLOB_INT],
            gf: vec![0.0; obs::GLOB_FLOAT],
            oa: vec![0; obs::MAX_OPS * obs::OP_INT],
            of: vec![0.0; obs::MAX_OPS * obs::OP_FLOAT],
            n_ops: 0,
            n_tokens: 0,
            n_hand: 0,
        }
    }

    /// A copy of `r`.
    pub fn of(r: &Rows<'_>) -> ObsRow {
        ObsRow {
            ti: r.ti.to_vec(),
            tf: r.tf.to_vec(),
            gi: r.gi.to_vec(),
            gf: r.gf.to_vec(),
            oa: r.oa.to_vec(),
            of: r.of.to_vec(),
            n_ops: *r.n_ops,
            n_tokens: *r.n_tokens,
            n_hand: *r.n_hand,
        }
    }

    pub fn rows(&mut self) -> Rows<'_> {
        Rows {
            ti: &mut self.ti,
            tf: &mut self.tf,
            gi: &mut self.gi,
            gf: &mut self.gf,
            oa: &mut self.oa,
            of: &mut self.of,
            n_ops: &mut self.n_ops,
            n_tokens: &mut self.n_tokens,
            n_hand: &mut self.n_hand,
        }
    }
}

/// `ValueError` unless `n` scored play candidates can be requested.
pub(crate) fn check_candidates(n: usize) -> PyResult<()> {
    if n > obs::MAX_CANDIDATES {
        return Err(PyValueError::new_err(format!("play_candidates must be at most {}, got {n}", obs::MAX_CANDIDATES)));
    }
    Ok(())
}
