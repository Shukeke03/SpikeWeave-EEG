# Project instructions

- Current phase includes real data construction. The user authorized public EEG downloads, mirror acceleration and preprocessing through `ssh legion`, alongside local coding and private GitHub synchronization. Paid training jobs are separate later steps.
- Preserve waveform, wavelet-structure and clinical-text branches, three-way representation alignment, and explicit wavelet-conditioned diffusion in any architecture revision.
- Keep Sketch2CT and MSADN as accurately attributed inspirations. JET, GenEEG, EEG-CLIP and ELM are relevant prior work; architectural combinations are not automatically novel.
- Distinguish a proposed mechanism, a published result and a result measured in this repository. Read docs/engineering-validation.md for measured engineering checks; clinical performance remains unmeasured.
- Keep generation-time transport/noise time separate from physical EEG time.
- Target EEG, full target CWT and clean target embeddings are teachers/targets only in externally conditioned generation. Optional reference EEG must be from an allowed non-target support sample.
- Preserve physical units, montage and channel provenance. A bipolar lead is an electrode pair, not a single scalp electrode.
- Perform patient/event/record splits before windowing and statistical fitting. Never use locked test EEG to fit normalization or construct adaptation support.
- Do not commit source clinical data, reports, credentials or weights. Check third-party code and model licenses before copying them; paper access does not establish code reuse permission.
- Verify privacy of the remote repository after creation or settings changes.
