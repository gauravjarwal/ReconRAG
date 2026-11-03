import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from evals.label import Labels, agreement_report, make_handler


@pytest.fixture
def labels(tmp_path):
    queue = tmp_path / "queue.jsonl"
    rows = [
        {
            "id": f"human-{i}",
            "input": {"question": "q", "context": "c", "answer": "a"},
            "source_id": f"judge-{i}",
        }
        for i in range(3)
    ]
    queue.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Labels(queue, tmp_path / "labels.jsonl")


def test_starts_at_the_first_item(labels):
    assert labels.state()["position"] == 0


def test_saving_a_label_persists_and_moves_on(labels):
    labels.save("human-0", True)
    reopened = Labels(labels.queue_path, labels.labels_path)
    assert reopened.state()["position"] == 1
    assert reopened.state()["done"] == 1


def test_changing_a_label_keeps_only_the_latest(labels):
    labels.save("human-0", True)
    labels.save("human-0", False)
    assert labels.labels() == {"human-0": False}


def test_unknown_ids_are_rejected(labels):
    with pytest.raises(KeyError):
        labels.save("nope", True)


def test_agreement_is_unavailable_until_labels_exist(labels):
    assert agreement_report(labels, seeded={})["items"] == 0


def test_agreement_compares_human_labels_with_seeded_truth(labels):
    labels.save("human-0", True)
    labels.save("human-1", False)
    report = agreement_report(labels, seeded={"judge-0": True, "judge-1": True})
    assert report["items"] == 2
    assert report["accuracy"] == 0.5


@pytest.fixture
def server(labels):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(labels))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body, content_type="application/json"):
    request = urllib.request.Request(
        f"{url}/label", data=json.dumps(body).encode(), headers={"Content-Type": content_type}
    )
    try:
        return urllib.request.urlopen(request, timeout=5).status
    except urllib.error.HTTPError as error:
        return error.code


def test_server_saves_a_json_label(server, labels):
    assert _post(server, {"id": "human-0", "faithful": True}) == 200
    assert labels.labels() == {"human-0": True}


def test_server_rejects_a_cross_site_style_form_post(server, labels):
    assert _post(server, {"id": "human-0", "faithful": True}, "text/plain") == 415
    assert labels.labels() == {}


def test_server_rejects_a_non_boolean_label(server, labels):
    assert _post(server, {"id": "human-0", "faithful": "yes"}) == 400
    assert labels.labels() == {}
