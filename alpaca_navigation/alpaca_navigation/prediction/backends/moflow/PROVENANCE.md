# Vendored MoFlow subset

`models/` and `utils/` copied from https://github.com/DSL-Lab/MoFlow at commit
`33811c3`, plus local modifications made for this project that were never
pushed upstream (8 files, +71/-33 at the time of vendoring).

Only these two directories are needed for inference; the upstream `data/`
(281 MB of datasets), training entry points and configs are omitted. The
internal layout is preserved because the package uses relative imports
(`from ..utils`, `from .context_encoder`).

Upstream license: MIT, see LICENSE.
