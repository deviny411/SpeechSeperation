# SpeechSeperation

Separating a doctor and a patient from two microphone recordings of a clinical session with heavy crosstalk and muddy audio. Built as a take-home, on a single free-tier Colab T4, with no model training.

## The problem

Two mics, two speakers, and each mic hears both of them. A pretrained separator (MossFormer2) splits each recording into two tracks, but it does not know which track is the doctor. Worse, it swaps speakers partway through a session, so a track can start as the doctor and end as the patient.

## Approach

I spent the first part understanding the signal, then used what I saw to fix the model's failure.

1. **Align and clean** (`CleanFinal.ipynb`)
   - Align the two mics by cross-correlation so they are sample-synchronized.
   - Band-pass 80 to 7500 Hz, then spectral-gate denoise in 30 s chunks so long sessions fit in memory.
   - Plot raw against cleaned waveforms and spectrograms, and warn if the lengths mismatch, because everything after this depends on the mics being aligned.
2. **Make the hidden structure visible** (`SeperationMossGCCPHAT.ipynb`)
   - Compute GCC-PHAT, the time delay between the two mics, for every 30 ms window.
   - Plotted over time, the doctor and the patient form separate clusters in delay space. That is a speaker signature that does not depend on any model.
3. **Separate with MossFormer2** (`MossFormer2_SS_16K` via ClearVoice) and plot its two tracks over time, marking Track A and Track B differently. This is where the speaker swaps become visible.
4. **Fix the swaps with the spatial signature**
   - Track each speaker's delay cluster over time, with handling for uncertain windows and re-acquisition.
   - Label the doctor and patient lanes by matching spatial hits to the loud segments of the doctor mic.
   - Re-stitch the model's two tracks by speaker identity, and log every stitching decision.

## How I checked it

There is no clean reference recording, so I did not compute SI-SDR or WER. Instead I used the spatial signature of the raw mics as a proxy for ground truth:

- Overlay the clusters found in each separated track on the raw clusters, to confirm each track follows one speaker.
- Compare the final stitched tracks against the reference mic and check that each reproduces its speaker's cluster.
- Audit the stitch log around swap points, and listen to the results.

This shows the speakers are assigned consistently. It does not measure how clean the audio is, and I did not claim a score.

## Constraints

- One free-tier Colab T4 and a short time window.
- No training. A pretrained separator plus classical signal processing (cross-correlation, GCC-PHAT, spectral gating).
- Chunked processing so full sessions stay inside memory.

## How to run

This was done in Google Colab.

1. Run `CleanFinal.ipynb` first. It aligns and cleans each session and writes the cleaned tracks next to the originals.
2. Run `SeperationMossGCCPHAT.ipynb` to separate and stitch. Check the paths first.
3. Look at the plots and listen to the results to judge performance.

## Links

- Final file outputs: https://drive.google.com/drive/folders/1p_aHhCnaAOh-WF44XcDvW8O5Q6yR7lm7?usp=sharing
- Documented approach: https://docs.google.com/document/d/1_Gf_Am9_mbLGb_6pU4k5W8aXivyETiJy5yYzAO3mOu8/edit?usp=sharing
- Other files and tests: https://drive.google.com/drive/folders/16e63VQsayBiQuDbwyuxtAFa3qFGL69gT?usp=sharing
