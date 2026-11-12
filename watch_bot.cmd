@echo off
rem Opens Balatro and lets the best agent play (run26_main's SWA + round search + endgame search from ante 7; its ante-7/8 shops pause to think). Extra arguments pass through, e.g. --speed 4.
rem swa_main (frozen critic + 1 epoch on 4,096 envs, 2026-10-29): 80.3% greedy in sim vs 65.9% for the previous swa_e1 (12,288 paired games over 3 seed sets); its search-enabled win rate is not measured yet.
rem Do not click in the game while it plays; close the game window to stop.
cd /d "%~dp0"
uv run python -m antezero_agent.play_real agent\runs\run26_main\swa_main.pt --watch --round-search --endgame-search --endgame-ante 7 --device auto --seeds 1000 %*
pause
