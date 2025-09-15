import shutil
import tempfile

import pytest

from orcestra.ipfs import ipfs_add


@pytest.mark.skipif(not shutil.which("ipfs"), reason="IPFS not found")
def test_ipfs_add():
    fp = tempfile.NamedTemporaryFile()
    fp.write(b"Hello world!")

    assert (
        ipfs_add(fp.name)
        == "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
    )

    fp.close()
