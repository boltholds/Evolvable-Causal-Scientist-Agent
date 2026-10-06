from hashlib import sha256
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import re


SOURCE_URL = "https://vreeken.groups.cispa.de/prj/vario/vario-v20220815.zip"
SOURCE_SHA256 = "67df25a256622f86fe7c7b469e2928ddcd2252680b18502699786041ca554652"


def test_probe_official_vario_api_contract() -> None:
    data = urlopen(SOURCE_URL, timeout=90).read()
    assert sha256(data).hexdigest() == SOURCE_SHA256
    wanted_suffixes = (
        "/examples/example_Vario_on_target.R",
        "/R/vario_pi.R",
        "/R/init.R",
        "/R/mdl.R",
        "/R/cls_pi_search.R",
        "/R/utils/utils_regression.R",
    )
    with ZipFile(BytesIO(data)) as archive:
        files = {
            name: archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if "__MACOSX" not in name
            and name.endswith(wanted_suffixes)
        }
    dependencies = sorted({
        match.group(1)
        for text in files.values()
        for match in re.finditer(
            r"(?:library|require)\s*\(\s*['\"]?([A-Za-z0-9_.]+)",
            text,
        )
    })
    namespace_uses = sorted({
        match.group(1)
        for text in files.values()
        for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_.]+)::", text)
    })
    raise AssertionError(
        {
            "dependencies": dependencies,
            "namespace_uses": namespace_uses,
            "files": files,
        }
    )
