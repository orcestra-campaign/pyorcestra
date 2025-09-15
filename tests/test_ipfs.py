import tempfile

from orcestra.ipfs import ipfs_add


def test_ipfs_add():
    fp = tempfile.NamedTemporaryFile()
    fp.write(b"Hello world!")

    assert (
        ipfs_add(fp.name)
        == "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
    )

    fp.close()
