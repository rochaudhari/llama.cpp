import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


# without unified KV, an idle slot that has not changed is copied to cache-ram once, not again after its copy was evicted
def test_unchanged_idle_slot_copied_once():
    global server
    server.n_slots = 3
    server.n_ctx = 3 * 2048
    server.kv_unified = False
    server.cache_ram = 1 # MiB, room for one of the states below
    server.server_embeddings = True # embedding requests share the slots with completions
    server.start()
    log = LogReader(server.log_path)

    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    def request(path, data):
        res = server.make_request("POST", path, data=data)
        assert res.status_code == 200

    prompts = [f"{i}. " + LONG_PROMPT * 7 for i in range(3)]
    for i, prompt in enumerate(prompts):
        request("/completion", {"prompt": prompt, "id_slot": i})

    # launching slot 2 copied idle slot 1, which evicted the copy of idle slot 0
    assert "making room for prompt cache entry" in log.drain()

    # slots 0 and 1 have not changed since their copies were made - they are not copied again
    request("/completion", {"prompt": prompts[2] + " The knight finally reached the castle gates.", "id_slot": 2})
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # slot 0 is reused by a prompt that shares most of its conversation - its evicted copy is made again first
    request("/completion", {"prompt": prompts[0][:len(prompts[0]) * 4 // 5] + " Then the story took another turn.", "id_slot": 0})
    assert "__TEST_TAG_CACHE_REUSED_SLOT__" in log.drain()

    # an embedding request takes slot 1, the least recently used, and drops all of it - its evicted copy is made again first
    request("/embeddings", {"input": "The fishermen mended their nets."})
    assert "__TEST_TAG_CACHE_REUSED_SLOT__" in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()
