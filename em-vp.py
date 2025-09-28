"""Compatibility wrapper for the legacy `em-vp` module name.

The main implementation now lives in `em_vp.py`.  This shim re-exports all
public symbols so existing imports that refer to `em-vp` continue to work until
callers update their references.
"""

from em_vp import *  # noqa: F401,F403
