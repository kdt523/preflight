from apps.ingest.signature import sign, verify

BODY = b'{"action":"opened"}'


def test_valid_signature():
    assert verify("s3cret", BODY, sign("s3cret", BODY))


def test_rejects_wrong_secret_tampered_body_and_bad_headers():
    good = sign("s3cret", BODY)
    assert not verify("other", BODY, good)
    assert not verify("s3cret", BODY + b" ", good)
    assert not verify("s3cret", BODY, None)
    assert not verify("s3cret", BODY, good.replace("sha256=", "sha1="))


def test_empty_secret_rejects_everything():
    assert not verify("", BODY, sign("", BODY))
