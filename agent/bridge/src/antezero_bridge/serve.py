"""CLI alias: ``python -m antezero_bridge.serve [--check|--install|--serve]``
(see :mod:`antezero_bridge.launch`)."""

from .launch import main

if __name__ == "__main__":
    raise SystemExit(main())
