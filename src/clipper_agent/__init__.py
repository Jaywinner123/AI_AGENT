"""
Clipper Agent - Automated Content Clipping and Publishing System

A comprehensive system for discovering, extracting, editing, and publishing
short-form video clips from long-form content sources.
"""

__version__ = "0.1.0"
__author__ = "Clipper Agent Team"
__email__ = "contact@clipperagent.com"

from .core.clipper import ClipperAgent
from .config.settings import ClipperConfig

__all__ = ["ClipperAgent", "ClipperConfig"]