import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def test_frontend_exposes_only_the_two_cstr_environments():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    scenarios = re.findall(r'data-scenario="([^"]+)"', html)
    assert scenarios == ["cstr", "cstr_jacketed_p2"]
    assert 'data-mode="mpc"' not in html
    assert "challenge.html" not in html


def test_all_bundled_policy_urls_exist():
    source = (FRONTEND / "js/sim/controllers.js").read_text(encoding="utf-8")
    urls = re.findall(r"url: '(\./models/[^']+)'", source)
    assert len(urls) == 3
    for url in urls:
        assert (FRONTEND / url.removeprefix("./")).is_file()


def test_model_directory_has_no_unlisted_variants():
    actual = {path.name for path in (FRONTEND / "models").glob("*.onnx")}
    assert actual == {
        "rlpd_cstr_upstream.onnx",
        "rlpd_cstr_jacketed_p24_official.onnx",
        "rlpd_cstr_jacketed_p24_field_v1.onnx",
    }
