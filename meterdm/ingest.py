from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

from meterdm.config import Config

USER_AGENT = "meterdm/0.1"
HTML_MARKERS = (b"<!doctype", b"<html", b"<?xml")
SHEET_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EXCEL_EPOCH = datetime(1899, 12, 30)
EXTRACTORS = ("tar", "7z", "unzip")
HALFHOURLY = "halfhourly"
TARIFF_COLUMNS = ("TariffDateTime", "Tariff")
HOUSEHOLDS = "households"
DAILY = "daily"
TARIFFS = "tariffs"


class DownloadError(RuntimeError):
    pass


class IncompleteTransfer(RuntimeError):
    pass


def _request(url, headers=None):
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged)


def _looks_like_html(payload: bytes) -> bool:
    head = payload[:512].lstrip().lower()
    return any(head.startswith(marker) for marker in HTML_MARKERS)


def probe(url: str, config: Config) -> dict:
    report = {"url": url, "ok": False, "status": None, "bytes": None, "content_type": None, "error": None}
    try:
        with urllib.request.urlopen(
            _request(url, {"Range": "bytes=0-4095"}), timeout=config.download.timeout_seconds
        ) as response:
            payload = response.read(4096)
            report["status"] = response.status
            report["content_type"] = response.headers.get("Content-Type")
            length = response.headers.get("Content-Range") or response.headers.get("Content-Length")
            report["bytes"] = length
            if config.download.reject_html and _looks_like_html(payload):
                report["error"] = "response body looks like an html page"
            else:
                report["ok"] = True
    except urllib.error.HTTPError as error:
        report["status"] = error.code
        report["error"] = f"http {error.code}"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def probe_sources(config: Config) -> dict:
    targets = {
        "london_archive": config.london_source.archive_url,
        "london_mirror": config.london_source.mirror_url,
        "portugal_archive": config.portugal_source.archive_url,
        "portugal_mirror": config.portugal_source.mirror_url,
        "weather": _weather_url(config, config.fleet(config.enabled_fleets[0])),
    }
    return {name: probe(url, config) for name, url in targets.items()}


def _weather_url(config: Config, fleet) -> str:
    query = urllib.parse.urlencode(
        {
            "latitude": fleet.latitude,
            "longitude": fleet.longitude,
            "start_date": config.weather_source.start_date.isoformat(),
            "end_date": config.weather_source.end_date.isoformat(),
            "hourly": config.weather_source.hourly_variable,
            "timezone": "UTC",
        }
    )
    return f"{config.weather_source.archive_url}?{query}"


def download(url: str, target: Path, config: Config, progress=None) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    settings = config.download
    last_error = None
    for attempt in range(1, settings.retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        try:
            with urllib.request.urlopen(_request(url, headers), timeout=settings.timeout_seconds) as response:
                if offset and response.status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                expected = _expected_total(response, offset)
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(settings.chunk_bytes)
                        if not chunk:
                            break
                        handle.write(chunk)
            size = partial.stat().st_size
            if expected is not None and size != expected:
                raise IncompleteTransfer(f"{url} delivered {size} of {expected} bytes")
            if size < settings.min_bytes:
                raise DownloadError(f"{url} returned {size} bytes, below the {settings.min_bytes} byte floor")
            with partial.open("rb") as handle:
                head = handle.read(512)
            if settings.reject_html and _looks_like_html(head):
                partial.unlink(missing_ok=True)
                raise DownloadError(f"{url} returned an html page rather than a data file")
            partial.replace(target)
            digest = _sha256(target)
            if progress:
                progress(f"downloaded {target.name} bytes={size} sha256={digest[:12]}")
            return {"url": url, "path": str(target), "bytes": size, "sha256": digest, "attempts": attempt}
        except DownloadError:
            raise
        except Exception as error:
            last_error = error
            if attempt < settings.retries:
                time.sleep(settings.backoff_seconds * attempt)
    raise DownloadError(f"{url} failed after {settings.retries} attempts: {last_error}")


def _expected_total(response, offset):
    span = response.headers.get("Content-Range")
    if span and "/" in span:
        total = span.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    length = response.headers.get("Content-Length")
    if length and length.isdigit():
        return offset + int(length)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_member(header: str) -> str | None:
    lowered = header.lower()
    if "lclid" not in lowered:
        return None
    if "tariff" in lowered:
        return TARIFFS
    if "acorn" in lowered:
        return HOUSEHOLDS
    if "energy_sum" in lowered:
        return DAILY
    if "datetime" in lowered or "tstp" in lowered:
        return HALFHOURLY
    return None


def _destination(kind, name, base):
    if kind == HALFHOURLY:
        return base / HALFHOURLY / name
    if kind == HOUSEHOLDS:
        return base / "informations_households.csv"
    if kind == DAILY:
        return base / "daily_dataset.csv"
    return base / "tariffs.csv"


def _copy_stream(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as sink:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)


def readable_with_zipfile(archive: Path, members) -> bool:
    if not members:
        return False
    with zipfile.ZipFile(archive) as bundle, bundle.open(members[0]) as handle:
        try:
            handle.read(1)
        except NotImplementedError:
            return False
    return True


def external_extract(archive: Path, destination: Path, progress=None) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    commands = {
        "tar": ["tar", "-xf", str(archive), "-C", str(destination)],
        "7z": ["7z", "x", f"-o{destination}", "-y", str(archive)],
        "unzip": ["unzip", "-o", "-q", str(archive), "-d", str(destination)],
    }
    for name in EXTRACTORS:
        if shutil.which(name) is None:
            continue
        if progress:
            progress(f"unpacking {archive.name} with {name}")
        result = subprocess.run(commands[name], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return name
    raise DownloadError(
        f"{archive} uses a compression method python cannot read and none of "
        f"{', '.join(EXTRACTORS)} unpacked it"
    )


def extract_london(archive: Path, config: Config, progress=None) -> dict:
    base = config.paths.raw_dir / "london"
    (base / HALFHOURLY).mkdir(parents=True, exist_ok=True)
    found = {HALFHOURLY: 0, HOUSEHOLDS: 0, DAILY: 0, TARIFFS: 0, "skipped": 0}
    with zipfile.ZipFile(archive) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if not name.endswith("/") and name.lower().endswith((".csv", ".txt"))
        ]
    if readable_with_zipfile(archive, members):
        with zipfile.ZipFile(archive) as bundle:
            for member in members:
                with bundle.open(member) as handle:
                    kind = classify_member(handle.readline().decode("utf-8", "replace"))
                if kind is None:
                    found["skipped"] += 1
                    continue
                with bundle.open(member) as source:
                    _copy_stream(source, _destination(kind, Path(member).name, base))
                found[kind] += 1
                if progress and found[HALFHOURLY] and found[HALFHOURLY] % 20 == 0:
                    progress(f"extracted {found[HALFHOURLY]} meter files")
    else:
        staging = config.paths.raw_dir / "_extract_london"
        shutil.rmtree(staging, ignore_errors=True)
        tool = external_extract(archive, staging, progress)
        if progress:
            progress(f"{tool} finished, classifying members")
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in (".csv", ".txt"):
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                kind = classify_member(handle.readline())
            if kind is None:
                found["skipped"] += 1
                continue
            target = _destination(kind, path.name, base)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            found[kind] += 1
        shutil.rmtree(staging, ignore_errors=True)
    if found[HALFHOURLY] == 0:
        raise DownloadError(f"{archive} contained no half hourly meter files")
    if progress:
        progress(f"london archive gave {found[HALFHOURLY]} meter files")
    return found


def _shared_strings(book):
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{SHEET_NAMESPACE}t"))
        for item in root.findall(f"{SHEET_NAMESPACE}si")
    ]


def read_tariff_workbook(path: Path):
    with zipfile.ZipFile(path) as book:
        strings = _shared_strings(book)
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{SHEET_NAMESPACE}row"):
        cells = {}
        for cell in row.findall(f"{SHEET_NAMESPACE}c"):
            reference = cell.get("r") or ""
            column = "".join(character for character in reference if character.isalpha())
            holder = cell.find(f"{SHEET_NAMESPACE}v")
            if holder is None or holder.text is None:
                continue
            if cell.get("t") == "s":
                cells[column] = strings[int(holder.text)]
            else:
                cells[column] = holder.text
        if "A" not in cells or "B" not in cells:
            continue
        try:
            serial = float(cells["A"])
        except ValueError:
            continue
        moment = EXCEL_EPOCH + timedelta(days=serial) + timedelta(seconds=30)
        rows.append((moment.replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S"), cells["B"]))
    if not rows:
        raise DownloadError(f"{path} held no dated tariff rows")
    return rows


def fetch_tariffs(config: Config, progress=None) -> dict:
    target = config.paths.raw_dir / "archives" / "tariffs.xlsx"
    result = download(config.london_source.tariff_url, target, config, progress)
    rows = read_tariff_workbook(target)
    destination = config.paths.raw_dir / "london" / "tariffs.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([f"{TARIFF_COLUMNS[0]},{TARIFF_COLUMNS[1]}"] + [f"{a},{b}" for a, b in rows])
    destination.write_text(body + "\n", encoding="ascii")
    result["rows"] = len(rows)
    result["labels"] = sorted({label for _, label in rows})
    if progress:
        progress(f"tariff schedule rows={len(rows)} labels={result['labels']}")
    return result


def extract_portugal(archive: Path, config: Config, progress=None) -> dict:
    destination = config.paths.raw_dir / "portugal" / config.portugal_source.member
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.lower().endswith((".txt", ".csv"))]
        if not members:
            raise DownloadError(f"{archive} contained no text member")
        wanted = next((name for name in members if Path(name).name == config.portugal_source.member), members[0])
        with bundle.open(wanted) as source, destination.open("wb") as sink:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                sink.write(chunk)
    if progress:
        progress(f"extracted {destination.name} bytes={destination.stat().st_size}")
    return {"member": wanted, "bytes": destination.stat().st_size}


def fetch_weather(config: Config, progress=None) -> dict:
    results = {}
    for name in config.enabled_fleets:
        fleet = config.fleet(name)
        target = config.paths.raw_dir / "weather" / f"{name}.json"
        results[name] = download(_weather_url(config, fleet), target, config, progress)
        payload = json.loads(target.read_text())
        if "hourly" not in payload or "temperature_2m" not in payload.get("hourly", {}):
            raise DownloadError(f"weather response for {name} has no hourly temperature series")
        results[name]["hours"] = len(payload["hourly"]["temperature_2m"])
    return results


def fetch_all(config: Config, progress=None) -> dict:
    manifest = {"weather": fetch_weather(config, progress)}
    if config.fleet("london").enabled:
        archive = config.paths.raw_dir / "archives" / "london.zip"
        try:
            manifest["london_download"] = download(config.london_source.archive_url, archive, config, progress)
        except DownloadError:
            manifest["london_download"] = download(config.london_source.mirror_url, archive, config, progress)
        manifest["london_extract"] = extract_london(archive, config, progress)
        manifest["london_tariffs"] = fetch_tariffs(config, progress)
    if config.fleet("portugal").enabled:
        archive = config.paths.raw_dir / "archives" / "portugal.zip"
        try:
            manifest["portugal_download"] = download(config.portugal_source.archive_url, archive, config, progress)
        except DownloadError:
            manifest["portugal_download"] = download(config.portugal_source.mirror_url, archive, config, progress)
        manifest["portugal_extract"] = extract_portugal(archive, config, progress)
    path = config.paths.raw_dir / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="ascii")
    return manifest
