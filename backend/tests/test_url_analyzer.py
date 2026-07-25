"""Tests for URL extraction and risk scoring."""

from __future__ import annotations

from url_analyzer import analyze_urls


def test_public_suffix_domain_is_resolved_correctly() -> None:
    result = analyze_urls("Apply at https://jobs.seek.com.au/software-engineer")
    assert result["urls"][0]["domain"] == "seek.com.au"
    assert "recruiting_lure" not in result["urls"][0]["flag_codes"]


def test_duplicate_urls_are_analyzed_once() -> None:
    result = analyze_urls("See https://example.com and https://example.com.")
    assert result["urls_found"] == 1


def test_shortener_and_non_https_are_flagged() -> None:
    result = analyze_urls("Apply at http://bit.ly/special-offer")
    flags = result["urls"][0]["flag_codes"]
    assert "shortener" in flags
    assert "non_https" in flags


def test_direct_ip_address_is_flagged() -> None:
    result = analyze_urls("Open http://192.0.2.1/apply")
    assert "ip_address" in result["urls"][0]["flag_codes"]
