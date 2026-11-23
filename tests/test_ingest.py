from __future__ import annotations

import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import apply_tiny, write_config

from meterdm import ingest
from meterdm.config import load_config

PAYLOAD = b"meter,reading\n" + b"".join(f"MAC{index:06d},0.5\n".encode("ascii") for index in range(4000))
HTML = b"<!DOCTYPE html>\n<html><body>not found</body>" + b"<p>padding</p>" * 200 + b"</html>\n"


class Handler(BaseHTTPRequestHandler):
    state = {"truncate_once": True, "fail_count": 0}

    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path.startswith("/html"):
            self._send(HTML, 200, "text/html")
            return
        if self.path.startswith("/tiny"):
            self._send(b"x", 200, "text/plain")
            return
        if self.path.startswith("/flaky"):
            Handler.state["fail_count"] += 1
            if Handler.state["fail_count"] < 3:
                self.send_response(503)
                self.end_headers()
                return
            self._send(PAYLOAD, 200, "text/csv")
            return
        if self.path.startswith("/resume"):
            start = 0
            header = self.headers.get("Range")
            if header:
                start = int(header.split("=")[1].split("-")[0])
            body = PAYLOAD[start:]
            if Handler.state["truncate_once"] and not header:
                Handler.state["truncate_once"] = False
                body = body[: len(PAYLOAD) // 3]
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.send_header("Content-Type", "text/csv")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(206 if header else 200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "text/csv")
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(PAYLOAD, 200, "text/csv")

    def _send(self, body, status, content_type):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def config(tmp_path):
    return load_config(write_config(tmp_path / "project", apply_tiny))


def test_download_writes_the_file_and_records_a_digest(server, config, tmp_path):
    target = tmp_path / "out" / "plain.csv"
    result = ingest.download(f"{server}/plain", target, config)
    assert target.read_bytes() == PAYLOAD
    assert result["bytes"] == len(PAYLOAD)
    assert len(result["sha256"]) == 64
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_download_rejects_an_html_error_page(server, config, tmp_path):
    with pytest.raises(ingest.DownloadError, match="html page"):
        ingest.download(f"{server}/html", tmp_path / "page.csv", config)
    assert not (tmp_path / "page.csv").exists()


def test_download_rejects_a_response_below_the_floor(server, config, tmp_path):
    with pytest.raises(ingest.DownloadError, match="below the"):
        ingest.download(f"{server}/tiny", tmp_path / "tiny.csv", config)


def test_download_retries_a_failing_server(server, config, tmp_path):
    Handler.state["fail_count"] = 0
    result = ingest.download(f"{server}/flaky", tmp_path / "flaky.csv", config)
    assert result["attempts"] == 3
    assert (tmp_path / "flaky.csv").read_bytes() == PAYLOAD


def test_download_resumes_a_truncated_transfer(server, config, tmp_path):
    Handler.state["truncate_once"] = True
    target = tmp_path / "resume.csv"
    result = ingest.download(f"{server}/resume", target, config)
    assert target.read_bytes() == PAYLOAD
    assert result["attempts"] == 2


def test_probe_reports_html_and_success(server, config):
    good = ingest.probe(f"{server}/plain", config)
    bad = ingest.probe(f"{server}/html", config)
    unreachable = ingest.probe("https://127.0.0.1:1/blocked", config)
    assert good["ok"] is True
    assert bad["ok"] is False and "html" in bad["error"]
    assert unreachable["ok"] is False and unreachable["error"]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("LCLid,stdorToU,DateTime,KWH/hh (per half hour) ", ingest.HALFHOURLY),
        ("LCLid,tstp,energy(kWh/hh)", ingest.HALFHOURLY),
        ("LCLid,stdorToU,Acorn,Acorn_grouped,file", ingest.HOUSEHOLDS),
        ("LCLid,day,energy_median,energy_sum,energy_count", ingest.DAILY),
        ("LCLid,TariffDateTime,Tariff", ingest.TARIFFS),
        ("station,date,tmax", None),
    ],
)
def test_member_classification(header, expected):
    assert ingest.classify_member(header) == expected


def test_extract_london_sorts_members_by_header(config, tmp_path):
    archive = tmp_path / "london.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a/block_0.csv", "LCLid,stdorToU,DateTime,KWH/hh (per half hour) \nMAC000002,Std,x,0.1\n")
        bundle.writestr("a/block_1.csv", "LCLid,stdorToU,DateTime,KWH/hh (per half hour) \nMAC000003,Std,x,0.2\n")
        bundle.writestr(
            "a/informations_households.csv", "LCLid,stdorToU,Acorn,Acorn_grouped,file\nMAC000002,Std,A,B,c\n"
        )
        bundle.writestr("a/daily.csv", "LCLid,day,energy_sum,energy_count\nMAC000002,2012-01-01,1.0,48\n")
        bundle.writestr("a/notes.txt", "nothing useful\n")
    found = ingest.extract_london(archive, config)
    assert found[ingest.HALFHOURLY] == 2
    assert found[ingest.HOUSEHOLDS] == 1
    assert found[ingest.DAILY] == 1
    assert (config.paths.raw_dir / "london" / "halfhourly" / "block_1.csv").is_file()
    assert (config.paths.raw_dir / "london" / "informations_households.csv").is_file()


def test_extract_london_rejects_an_archive_without_meter_files(config, tmp_path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("readme.txt", "no data here")
    with pytest.raises(ingest.DownloadError, match="no half hourly"):
        ingest.extract_london(archive, config)


def test_extract_portugal_picks_the_named_member(config, tmp_path):
    archive = tmp_path / "portugal.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("other.txt", "junk")
        bundle.writestr(config.portugal_source.member, '"";"MT_001"\n"2011-01-01 00:15:00";0,5\n')
    result = ingest.extract_portugal(archive, config)
    assert result["member"].endswith(config.portugal_source.member)
    written = config.paths.raw_dir / "portugal" / config.portugal_source.member
    assert "MT_001" in written.read_text()


def test_weather_url_carries_the_fleet_location(config):
    url = ingest._weather_url(config, config.fleet("london"))
    assert "latitude=51.4775" in url
    assert "timezone=UTC" in url
    assert config.weather_source.start_date.isoformat() in url


def test_fetch_weather_validates_the_payload(server, config, monkeypatch):
    def fake(url, target, cfg, progress=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"hourly": {"time": ["2012-01-01T00:00"], "temperature_2m": [4.5]}}))
        return {"url": url, "path": str(target), "bytes": 10, "sha256": "0" * 64, "attempts": 1}

    monkeypatch.setattr(ingest, "download", fake)
    result = ingest.fetch_weather(config)
    assert result["london"]["hours"] == 1

    def broken(url, target, cfg, progress=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"error": True}))
        return {"url": url, "path": str(target), "bytes": 10, "sha256": "0" * 64, "attempts": 1}

    monkeypatch.setattr(ingest, "download", broken)
    with pytest.raises(ingest.DownloadError, match="no hourly temperature"):
        ingest.fetch_weather(config)


def test_zip_helper_reads_headers_without_full_extraction(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("x.csv", "LCLid,tstp,energy(kWh/hh)\n")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as bundle, bundle.open("x.csv") as handle:
        assert ingest.classify_member(handle.readline().decode()) == ingest.HALFHOURLY


TARIFF_SHARED = (
    '<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    "<si><t>TariffDateTime</t></si><si><t>Tariff</t></si><si><t>Normal</t></si><si><t>High</t></si></sst>"
)
TARIFF_SHEET = (
    '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    "<sheetData>"
    '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
    '<row r="2"><c r="A2" s="1"><v>41275</v></c><c r="B2" t="s"><v>2</v></c></row>'
    '<row r="3"><c r="A3" s="1"><v>41275.0208333333</v></c><c r="B3" t="s"><v>3</v></c></row>'
    "</sheetData></worksheet>"
)


def write_workbook(path, sheet=TARIFF_SHEET, shared=TARIFF_SHARED):
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("xl/sharedStrings.xml", shared)
        book.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


def test_tariff_workbook_converts_excel_serials(tmp_path):
    rows = ingest.read_tariff_workbook(write_workbook(tmp_path / "tariffs.xlsx"))
    assert rows == [("2013-01-01 00:00:00", "Normal"), ("2013-01-01 00:30:00", "High")]


def test_tariff_workbook_without_dates_is_rejected(tmp_path):
    empty = (
        '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row></sheetData></worksheet>'
    )
    with pytest.raises(ingest.DownloadError, match="no dated tariff rows"):
        ingest.read_tariff_workbook(write_workbook(tmp_path / "bad.xlsx", sheet=empty))


def test_fetch_tariffs_writes_a_csv(config, tmp_path, monkeypatch):
    def fake(url, target, cfg, progress=None):
        target.parent.mkdir(parents=True, exist_ok=True)
        write_workbook(target)
        return {"url": url, "path": str(target), "bytes": 1, "sha256": "0" * 64, "attempts": 1}

    monkeypatch.setattr(ingest, "download", fake)
    result = ingest.fetch_tariffs(config)
    written = (config.paths.raw_dir / "london" / "tariffs.csv").read_text()
    assert result["rows"] == 2
    assert result["labels"] == ["High", "Normal"]
    assert written.splitlines()[0] == "TariffDateTime,Tariff"
    assert "2013-01-01 00:30:00,High" in written


def test_unsupported_compression_falls_back_to_a_system_extractor(config, tmp_path, monkeypatch):
    archive = tmp_path / "london.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a/block_0.csv", "LCLid,stdorToU,DateTime,KWH/hh\nMAC000002,Std,x,0.1\n")

    monkeypatch.setattr(ingest, "readable_with_zipfile", lambda *args: False)
    calls = []

    def fake_extract(source, destination, progress=None):
        calls.append(source)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as bundle:
            bundle.extractall(destination)
        return "tar"

    monkeypatch.setattr(ingest, "external_extract", fake_extract)
    found = ingest.extract_london(archive, config)
    assert calls == [archive]
    assert found[ingest.HALFHOURLY] == 1
    assert (config.paths.raw_dir / "london" / "halfhourly" / "block_0.csv").is_file()
    assert not (config.paths.raw_dir / "_extract_london").exists()


def test_external_extract_reports_when_no_tool_exists(config, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    with pytest.raises(ingest.DownloadError, match="none of"):
        ingest.external_extract(tmp_path / "x.zip", tmp_path / "out")
