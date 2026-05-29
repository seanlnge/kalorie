"""Autonomous live-trading executor for Kalorie2.

This package consumes model trade signals and places Kalshi limit orders behind
hard, fail-closed risk controls. The read-only poller and web workstation never
import or trigger anything here; the only live-order path lives in this package
and is gated by explicit configuration.
"""
