import subprocess


def ipfs_add(path):
    """Add a path to IPFS recursively and return the CID.

    The IPFS options follow the ORCESTRA recommendations [0].

    [0] https://orcestra-campaign.org/ipfs.html
    """
    ret = subprocess.run(
        [
            "ipfs",
            "add",
            "--recursive",
            "--hidden",
            "--raw-leaves",
            "--chunker=size-1048576",
            "--quieter",
            path,
        ],
        capture_output=True,
    )

    if ret.returncode > 0:
        raise RuntimeError(ret.stderr)

    return ret.stdout.decode().strip()
