"""HTTP service over the pipeline.

Serves existing episodes, lets a user trigger a full run, and exposes a
personalization path (edit the brief, re-generate audio). An APScheduler
background job auto-runs a fresh episode on a configurable cron (default:
every Monday 09:00).
"""
