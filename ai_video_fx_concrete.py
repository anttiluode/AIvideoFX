#!/usr/bin/env python3
"""Launch AI Video FX with the Concrete adaptive diffusion mode registered."""

# Import before ai_video_fx pulls the fx_core registries into its GUI module.
import fx_concrete_refresh  # noqa: F401

from ai_video_fx import main


if __name__ == "__main__":
    main()
