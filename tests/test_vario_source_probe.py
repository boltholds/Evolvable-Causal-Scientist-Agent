from hashlib import sha256
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


SOURCE_URL = "https://vreeken.groups.cispa.de/prj/vario/vario-v20220815.zip"


def test_probe_official_vario_source_contract() -> None:
    data = urlopen(SOURCE_URL, timeout=90).read()
    with ZipFile(BytesIO(data)) as archive:
        names = [
            name for name in archive.namelist()
            if "__MACOSX" not in name and not name.endswith("/")
        ]
        text_files = {}
        for name in names:
            base = name.rsplit("/", 1)[-1]
            if (
                base in {"README", "DESCRIPTION", "NAMESPACE"}
                or "/R/" in name
            ):
                try:
                    text_files[name] = archive.read(name).decode("utf-8")
                except UnicodeDecodeError:
                    pass

    raise AssertionError(
        {
            "sha256": sha256(data).hexdigest(),
            "files": names,
            "selected_text": text_files,
        }
    )
