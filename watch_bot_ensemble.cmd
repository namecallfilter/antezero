@echo off
rem Opens Balatro and lets the 3-model ensemble play (swa_e + swa_h + swa_f, round search + endgame search from ante 7).
rem SUPERSEDED 2026-10-26: watch_bot.cmd now plays swa_e1, which is stronger than this ensemble (76.8% vs 73.6% on the same 1,024 sim games) and pauses less.
rem (Was about +1.9 pts over the old swa_e watch_bot in sim, 2,048 paired games, p=0.055, with ~2.2x longer search pauses.) Extra arguments pass through, e.g. --speed 4.
rem Do not click in the game while it plays; close the game window to stop.
cd /d "%~dp0"
uv run python -m antezero_agent.play_real agent\runs\run13\swa_e_u10000-u12400.pt+agent\runs\ensemble\swa_h.pt+agent\runs\ensemble\swa_f.pt --watch --round-search --endgame-search --endgame-ante 7 --device auto --seeds 1000 %*
pause
