"""Fetch a pinned Shakespeare corpus and verify its bytes before writing."""

import hashlib
import urllib.request
from pathlib import Path

REVISION = "370cbcd448eb7daf32f21a6be560b70e0b33c4e3"
URL = (
    f"https://raw.githubusercontent.com/karpathy/char-rnn/{REVISION}/data/tinyshakespeare/input.txt"
)
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("Downloaded corpus does not match the pinned checksum")
    path = Path("data/shakespeare.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(f"Wrote {len(content):,} verified bytes to {path}")


if __name__ == "__main__":
    main()
