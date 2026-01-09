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


@pytest.mark.skipif(not shutil.which("ipfs"), reason="IPFS not found")
def test_ipfs_add_cidv1():
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/world.txt", "w") as f:
            f.write("Hello world!")

        print(ipfs_add(temp_dir))
        assert (
            ipfs_add(temp_dir)
            == "bafybeidxxwpdugrw6ykgok6exrfx6ccdgioeixknswb5ocuhbdep2bdshy"
        )
