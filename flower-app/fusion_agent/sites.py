"""Allowlisted local services and an authenticated remote-site transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import evidence, heating

OPERATIONS = {
    "list_studies",
    "inspect_study",
    "list_cases",
    "diagnose_heating",
    "validate_heating",
}


class LocalSite:
    def __init__(self, site_id: str, root: Path | None = None, demo: bool = False):
        if demo and site_id not in heating.PROFILES:
            raise ValueError("Unknown demo site")
        self.site_id, self.root, self.demo = site_id, root, demo

    def describe(self):
        return {
            "site": self.site_id,
            "transport": "local",
            "kind": "synthetic demo"
            if self.demo
            else "repository simulation artifacts",
            "capabilities": (
                ["list_cases", "diagnose_heating", "validate_heating"]
                if self.demo
                else ["list_studies", "inspect_study"]
            ),
        }

    def call(self, operation: str, arguments: dict):
        if operation not in self.describe()["capabilities"]:
            raise ValueError("Operation unavailable at this site")
        if operation == "list_cases":
            if arguments:
                raise ValueError("list_cases takes no arguments")
            result = {
                "cases": [
                    {
                        "case_id": "heating-response",
                        "description": "Synthetic heating-response investigation",
                    }
                ]
            }
        elif operation == "diagnose_heating":
            result = heating.diagnose(self.site_id, **arguments)
        elif operation == "validate_heating":
            result = heating.validate(self.site_id, **arguments)
        elif operation == "list_studies":
            if arguments:
                raise ValueError("list_studies takes no arguments")
            result = evidence.list_studies(self.root)
        else:
            result = evidence.inspect_study(self.root, **arguments)
        return {"site": self.site_id, **result}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward site credentials to another host.
        return None


class RemoteSite:
    def __init__(self, site_id: str, url: str, token_env: str):
        p = urlparse(url)
        if (
            p.username
            or p.password
            or p.query
            or p.fragment
            or p.path not in ("", "/")
            or not p.hostname
            or not (
                p.scheme == "https"
                or (
                    p.scheme == "http"
                    and p.hostname in ("127.0.0.1", "localhost", "::1")
                )
            )
        ):
            raise ValueError(
                "Remote site needs HTTPS, or loopback HTTP, and a base URL"
            )
        self.site_id, self.url, self.token_env = site_id, url.rstrip("/"), token_env

    def describe(self):
        return {
            "site": self.site_id,
            "transport": "remote",
            "kind": "operator-configured site; query to verify evidence type",
            "capabilities": sorted(OPERATIONS),
        }

    def call(self, operation: str, arguments: dict):
        if operation not in OPERATIONS:
            raise ValueError("Unknown site operation")
        token = os.environ.get(self.token_env)
        if not token:
            raise ValueError("Site credential is not configured")
        request = Request(
            self.url + "/analysis",
            method="POST",
            data=json.dumps({"operation": operation, "arguments": arguments}).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with build_opener(NoRedirect).open(request, timeout=15) as response:
                body = response.read(256_001)
            if len(body) > 256_000:
                raise ValueError("Site response exceeds analysis size limit")
            data = json.loads(body)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ValueError("Site unavailable or request rejected") from exc
        if not isinstance(data, dict) or data.get("site") != self.site_id:
            raise ValueError("Site response identity mismatch")
        return data


def load_sites(config: str | None = None) -> dict:
    # Configuration is operator-owned, never a model-controlled filesystem path.
    if not config:
        return {s: LocalSite(s, demo=True) for s in heating.PROFILES}
    path = Path(config).resolve()
    rows = json.loads(path.read_text())["sites"]
    result = {}
    for row in rows:
        site_id = row["id"]
        if not isinstance(site_id, str) or site_id in result:
            raise ValueError("Site IDs must be unique strings")
        if row["type"] == "demo":
            site = LocalSite(site_id, demo=True)
        elif row["type"] == "checkpoint":
            site = LocalSite(site_id, (path.parent / row["root"]).resolve())
        elif row["type"] == "remote":
            site = RemoteSite(site_id, row["url"], row["token_env"])
        else:
            raise ValueError("Unknown site adapter")
        result[site_id] = site
    if not 1 <= len(result) <= 20:
        raise ValueError("Configure between 1 and 20 sites")
    return result
