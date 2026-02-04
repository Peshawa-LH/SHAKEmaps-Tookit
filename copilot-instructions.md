# Copilot / AI Agent Instructions for SHAKEmaps Toolkit

## Quick orientation ✅
- Purpose: Toolkit for fetching, visualizing, and modeling ShakeMap data (USGS ShakeMaps, DYFI, ruptures, VS30, propagation). Core Python modules are under `modules/` and interactive workflows live in the Jupyter notebooks in the repo root (e.g., `SHAKEfetch.ipynb`, `SHAKEmapper.ipynb`, `SHAKEpropagate.ipynb`).
- Key modules to read first: `modules/SHAKEfetch.py`, `modules/SHAKEmapper.py`, `modules/SHAKEpropagate.py`, `modules/SHAKEtools.py`.

---

## Big-picture architecture & flows 💡
- Single-event scoping: `SHAKEfetch` is event-centric (one instance = one USGS event id). Multi-event runs are implemented by instantiating multiple `SHAKEfetch` objects.
- Fetch → Visualize → Model flow:
  1. Fetch USGS products (via `SHAKEfetch`) using `libcomcat` and the `getproduct` CLI. Downloads go under `export/SHAKEfetch/`.
  2. Visualize and analyze grids using `SHAKEmapper` (Cartopy + Rasterio if available).
  3. Propagate wavefronts with `SHAKEpropagate` which reads ShakeMap XML/rasters and writes to `export/SHAKEpropagate/<event_id>/<case>/`.
- Notebooks demonstrate typical pipelines and serve as runnable integration tests for manual verification.

---

## Important external dependencies & environment 🔗
- Conda env files with required packages:
  - `enviroment/SHAKEdev.yml` (general dev env including cartopy, rasterio, matplotlib, pandas, rasterio, etc.)
  - `enviroment/usgs-comcat.yml` (includes `usgs-libcomcat` / comcat tooling)
- CLI requirement: `getproduct` is invoked by `SHAKEfetch._run_download_commands` to download product files. Ensure `getproduct` is available on PATH when testing those parts.
- Cartopy is optional: modules check availability and fall back to Matplotlib if unavailable (`_HAS_CARTOPY`). Code should treat plotting as optional.

---

## Project-specific patterns & conventions 📐
- Docstring-first: modules have long top-level docstrings that include "Version:" and short usage examples. When changing public behavior, update the module version in that header.
- Output layout convention: exporters write under `export/<ModuleName>/...` (e.g., `export/SHAKEfetch/`, `export/SHAKEpropagate/`). Use `_ensure_dir` or `os.makedirs(..., exist_ok=True)`.
- Logging: modules often use module-specific loggers and avoid adding duplicate handlers (`logger = logging.getLogger("SHAKEpropagate")`, check `if not logger.handlers:` before adding handlers). Follow existing logging style instead of configuring root logging.
- Defensive checks for external data: e.g., `if not self.earthquake: logging.error(...); return` in `SHAKEfetch` methods — preserve this defensive pattern when adding downloads or parsing steps.
- Optional dependency handling: import in try/except and expose `_HAS_CARTOPY`-style flags — follow this same approach for other optional GIS libs.

---

## Quick commands / smoke-tests (examples) ▶️
- Create environment (conda):
  - conda env create -f enviroment/SHAKEdev.yml
  - or conda env create -f enviroment/usgs-comcat.yml (if focusing on ComCat work)
- Manual quick checks (run in Python REPL inside the env, or use Jupyter):
  - from modules.SHAKEfetch import SHAKEfetch
    sf = SHAKEfetch('us7000m9g4')
    sf.get_event_info(version='last')  # requires `getproduct` and network
  - from modules.SHAKEpropagate import SHAKEpropagate, Inputs
    sim = SHAKEpropagate(Inputs(shakemap_xml='path/to/grid.xml'))
    result = sim.run_scenario(case_name='test', export_all=False)
  - from modules.SHAKEmapper import SHAKEmapper
    mapper = SHAKEmapper(extent=[90.4,103.4,13.0,26.8])
    fig, ax = mapper.create_basemap()

Notes: network + `getproduct` may be required for fetch examples. If Cartopy is not installed, plotting will still work in a reduced form.

---

## How to extend code safely (practical rules for AI agents) 🔧
- Keep public APIs stable: these modules are used by notebooks and rely on argument names and return structures (many functions return dicts with keys like `'summary'`, `'out_dir'` etc.). If changing return formats, update all notebooks and examples.
- Tests: there are no unit tests currently. Prefer adding small, focused unit tests under a new `tests/` folder when adding or refactoring behavior-critical code (e.g., parsing, numerical outputs). Tests should avoid network/IO by mocking `libcomcat` and CLI calls (e.g., `subprocess.run` for `getproduct`).
- Version bump in docstring header: when adding features or changing behavior, increment the `Version:` line in the module's top-level docstring (follow existing pattern like `26.1`, `26.4`).
- Output directories: use `os.makedirs(..., exist_ok=True)` and return the path to written artifacts.

---

## Where to look for examples in this repo 🔎
- `modules/SHAKEfetch.py` — product inspection & download patterns; uses `libcomcat` and `getproduct`.
- `modules/SHAKEmapper.py` — raster extraction with `rasterio` and plotting conventions (discrete colormaps, extent handling).
- `modules/SHAKEpropagate.py` — dataclass-driven settings (`Inputs`, `Settings`) and scenario-based `run_scenario` pattern. Good examples of modular scenario overrides and logging.
- `modules/SHAKEtools.py` — utility functions (UTM parsing, unit conversions, scales) that are safe to reuse and unit-test.
- Notebooks in repo root — integration usage and plot examples that should be kept up to date when changing behavior.

---

## Brief do / don't checklist for agents ✅ / ❌
- ✅ Follow existing logging and optional dependency patterns.
- ✅ Update module docstring `Version:` when changing behavior.
- ✅ Add unit tests for parsing/transformations; mock network/CLI calls.
- ✅ Use `export/` subfolders for artifacts and return the path in function outputs.

- ❌ Don't change public signatures silently (notebooks rely on them).
- ❌ Don't assume Cartopy or `getproduct` are present; code must handle absence gracefully.

---

If anything is unclear or you'd like more coverage (tests, CI, or a CONTRIBUTING.md with exact local checks), tell me which area to expand and I will iterate. 🙌
