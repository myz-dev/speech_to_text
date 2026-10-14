//! Speech-to-text from 16 kHz mono samples.
//! Backends: whisper.cpp via whisper-rs (Metal, M1) and
//! Parakeet TDT via ONNX Runtime (M2). See [PLAN.md](../../../PLAN.md).

pub mod whisper;
