# az_run95_d384_l8 at update 11,900

The latest checkpoint of the GCP run az_run95_d384_l8 (`--preset run95 d_model=384 layers=8 heads=12`), from
`gs://smb-runs-80b2a3f6/antezero/runs/az_run95_d384_l8/last.pt` as of 2026-11-26 (steps 779,878,400, games 3,708,855).
It is the full trainer checkpoint (model, config, optimizer, trainer state, curriculum), so training can resume from it.
It is xz-compressed and split into two parts to stay under GitHub's 100 MB per-file limit.

Reassemble:

    cat last.pt.xz.part* | xz -d > last.pt
    sha256sum last.pt   # 06e896f762abe94dcbf8bad85b7185d5c6fdf93925c8b97548a7b44f7dd269a8

config.json is the run's config; best.json lists the best score per deck.
