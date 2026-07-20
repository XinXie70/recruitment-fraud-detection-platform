import re
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse


URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s<>()\"']+|\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s<>()\"']*)?"
)

TRAILING_PUNCTUATION = ".,;:!?)]}'\""
SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
}
SUSPICIOUS_TLDS = {
    "click",
    "country",
    "fit",
    "ga",
    "gq",
    "loan",
    "ml",
    "rest",
    "surf",
    "tk",
    "top",
    "work",
    "xyz",
}
SUSPICIOUS_PATH_KEYWORDS = {
    "bank",
    "fee",
    "gift-card",
    "giftcard",
    "payment",
    "telegram",
    "transfer",
    "verify",
    "whatsapp",
}
RECRUITING_DOMAIN_KEYWORDS = {
    "apply",
    "career",
    "careers",
    "hiring",
    "job",
    "jobs",
    "recruit",
    "recruitment",
}


@dataclass(frozen=True)
class UrlFlag:
    code: str
    message: str
    weight: float


def _normalize_candidate(candidate: str) -> str:
    cleaned = candidate.strip().strip(TRAILING_PUNCTUATION)
    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = f"https://{cleaned}"
    return cleaned


def _registered_domain(hostname: str) -> str:
    parts = hostname.lower().strip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname.lower()


def _risk_level(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def _is_ip_address(hostname: str) -> bool:
    try:
        ip_address(hostname)
        return True
    except ValueError:
        return False


def _unique_urls(text: str) -> list[str]:
    seen = set()
    urls = []
    for match in URL_PATTERN.finditer(text):
        normalized = _normalize_candidate(match.group(0))
        if normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def _flag_url(url: str) -> tuple[str, list[UrlFlag]]:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    registered_domain = _registered_domain(hostname)
    path = f"{parsed.path} {parsed.query}".lower()
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""
    flags: list[UrlFlag] = []

    if parsed.scheme != "https":
        flags.append(UrlFlag("non_https", "The link does not use HTTPS.", 0.2))

    if registered_domain in SHORTENER_DOMAINS:
        flags.append(UrlFlag("shortener", "The link uses a URL shortener.", 0.35))

    if tld in SUSPICIOUS_TLDS:
        flags.append(UrlFlag("suspicious_tld", f"The domain uses the .{tld} top-level domain.", 0.25))

    if _is_ip_address(hostname):
        flags.append(UrlFlag("ip_address", "The link points directly to an IP address.", 0.3))

    if hostname.startswith("xn--") or ".xn--" in hostname:
        flags.append(UrlFlag("idn_domain", "The domain uses punycode characters.", 0.25))

    if any(keyword in path for keyword in SUSPICIOUS_PATH_KEYWORDS):
        flags.append(UrlFlag("sensitive_path", "The link path contains payment or messaging keywords.", 0.25))

    domain_tokens = set(re.split(r"[-.]", hostname))
    if domain_tokens & RECRUITING_DOMAIN_KEYWORDS and registered_domain not in {
        "linkedin.com",
        "indeed.com",
        "seek.com.au",
        "greenhouse.io",
        "lever.co",
        "workdayjobs.com",
    }:
        flags.append(UrlFlag("recruiting_lure", "The domain uses hiring or application keywords.", 0.15))

    return registered_domain, flags


def analyze_urls(text: str) -> dict[str, Any]:
    urls = []
    reasons = []

    for normalized_url in _unique_urls(text):
        domain, flags = _flag_url(normalized_url)
        score = min(1.0, sum(flag.weight for flag in flags))
        level = _risk_level(score)
        urls.append(
            {
                "url": normalized_url,
                "domain": domain,
                "risk_score": round(score, 3),
                "risk_level": level,
                "flags": [flag.message for flag in flags],
                "flag_codes": [flag.code for flag in flags],
            }
        )

    urls.sort(key=lambda item: item["risk_score"], reverse=True)
    overall_score = max((item["risk_score"] for item in urls), default=0.0)
    high_risk_count = sum(1 for item in urls if item["risk_level"] == "high")
    medium_risk_count = sum(1 for item in urls if item["risk_level"] == "medium")

    if not urls:
        reasons.append("No URLs were found in this listing.")
    else:
        if high_risk_count:
            reasons.append(f"{high_risk_count} high-risk link(s) were found.")
        if medium_risk_count:
            reasons.append(f"{medium_risk_count} medium-risk link(s) should be reviewed.")
        if any("shortener" in item["flag_codes"] for item in urls):
            reasons.append("One or more links use URL shorteners.")
        if any("non_https" in item["flag_codes"] for item in urls):
            reasons.append("One or more links do not use HTTPS.")
        if any("suspicious_tld" in item["flag_codes"] for item in urls):
            reasons.append("One or more links use suspicious top-level domains.")
        if all(not item["flags"] for item in urls):
            reasons.append("No major URL risk signals were detected.")

    return {
        "urls_found": len(urls),
        "risk_score": round(overall_score, 3),
        "risk_level": _risk_level(overall_score),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "urls": urls,
        "reasons": reasons,
    }
