# Concrete Dream mode

Run:

```bat
run_concrete_refresh.bat
```

or:

```bat
python3.13 ai_video_fx_concrete.py --preset "Concrete Dream"
```

The ordinary AI Video FX GUI opens with one extra diffusion effect/preset: **Concrete Dream (adaptive)**.

## What it does

The stock `AI Dream (img2img)` asks the diffusion worker for a live style map at the configured AI rate. Concrete Dream changes only the scheduling contract while it is enabled:

```text
camera frame
    ↓
48×27 luminance certificate
    ↓
predicted visible keyframe change
    ├─ small  → REUSE old generated keyframe
    │           PhaseRail carries it with live motion
    ├─ large  → REFRESH: drop the style map
    │           existing diffusion worker generates a new keyframe
    └─ audit  → deliberately refresh a supposedly safe frame
                compare old carried output with new diffusion output
                update learned input→output sensitivity
```

Removing or disabling Concrete Dream removes its `style_concrete` request token and the stock live-diffusion scheduling resumes.

## First controls to touch

- **Refresh tolerance**: higher = fewer diffusion calls / more stale-looking risk. Start at `0.12`.
- **Audit chance**: checked only at the decision rate, not every camera frame. Start at `0.05`.
- **Between refreshes**: `phase` transports the generated keyframe with PhaseRail; `hold` is the boring attacker.
- **Cold input threshold**: used before enough expensive refreshes exist to learn a sensitivity gain. Lower = more conservative.
- **Max key age**: hard ceiling that eventually buys a new diffusion image even in a quiet scene.

## What the HUD means

`Concrete REUSE` means no new diffusion image was requested.

`Concrete REFRESH` means the certificate predicted enough visible change to buy one.

`Concrete AUDIT` means the scheduler thought reuse was safe but deliberately bought a diffusion answer anyway.

`obs` is the RMS difference between the carried image that would have remained visible and the new expensive diffusion image. Audits whose observed error exceeds the refresh tolerance count as `miss` and make future predictions more conservative.

## First useful comparison

Use the same webcam pose/prompt and compare:

1. stock `Dream Machine` / `AI Dream (img2img)` with live diffusion;
2. `Concrete Dream` with transport=`hold`;
3. `Concrete Dream` with transport=`phase`.

Watch the diffusion rate in the app HUD / AI status and the Concrete refresh counter. The mode only earns its keep if it materially lowers expensive style generations without the output looking frozen or wrong.
