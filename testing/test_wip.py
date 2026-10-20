"""Tests for ``ramantools.wipfile`` — loading WITec project files via the WITec SDK.

These tests need the sample project ``test-data/FLG42_532.wip`` (~260 MB,
not pushed to the remote, same policy as ``specmap.txt``). The whole module
is skipped if it is missing, so the test suite stays green on a bare
checkout. ``WITecSDK`` itself is a regular dependency of ramantools.

The sample file is a real measurement session, so expectations below are
pinned to its contents (object IDs, pixel counts, a few metadata values).
If the sample file is ever replaced, update the ``FLG42`` constants in one
place at the top.
"""

import sys  # For inserting the repo root onto sys.path
from pathlib import Path  # OS-independent path handling
import warnings  # Silence curve_fit warnings in the pipeline test

import numpy as np  # Numerical comparisons
import pytest  # Test framework
import xarray as xr  # Type checks on returned DataArrays

# Ensure the in-tree ramantools is importable without ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ramantools as rt  # noqa: E402  (import after sys.path tweak)

# Skip the whole module when the sample project is unavailable.
WIP_PATH = ROOT / "test-data" / "FLG42_532.wip"
pytestmark = pytest.mark.skipif(
    not WIP_PATH.exists(),
    reason="sample project test-data/FLG42_532.wip not present (not tracked in git)",
)

# ---------------------------------------------------------------------------
# Facts about the sample project, verified by hand with the SDK.
# ---------------------------------------------------------------------------
# Full-size map: FLG_42--Scan Piezo--010--Spec.Data 1
FLG42_MAP_ID = 188
FLG42_MAP_SHAPE = (1600, 140, 64)       # (ramanshift, pixel_y, pixel_x)
FLG42_MAP_SIZE = (32.0, 70.0)           # (size_x, size_y) in µm — 64 * 0.5, 140 * 0.5
# Single spectrum extracted in Witec Project from pixel x=31, y=33 of that map
FLG42_SPEC_ID = 200
FLG42_SPEC_PIXEL = (31, 33)
# User-drawn boolean mask on the same scan
FLG42_MASK_ID = 254
# Fit-result image ("Fit Lorentz: x0") on the same scan, units rel. 1/cm
FLG42_IMAGE_ID = 208
# Small background-subtracted map and the G-band sum image Witec computed from it
FLG42_SMALL_MAP_ID = 107
FLG42_GSUM_IMAGE_ID = 126
# Instrument metadata from the scan's Information text
FLG42_LASER_NM = 532.033
FLG42_ITIME_S = 5.0
FLG42_SAMPLENAME = "FLG_42"


@pytest.fixture(scope="module")
def wip():
    """Open the sample project once for the whole module (opening is lazy and fast)."""
    return rt.wipfile(str(WIP_PATH))


@pytest.fixture(scope="module")
def flg42_map(wip):
    """The full-size map extracted as a ramanmap, shared across tests."""
    return wip.ramanmap(FLG42_MAP_ID)


class TestWipfileInventory:
    """The container itself: opening, listing, lookup and error paths."""

    def test_public_import_paths(self):
        """wipfile is reachable from both the package and the legacy shim, as one object."""
        from ramantools.ramantools import wipfile as shim_wipfile
        assert rt.wipfile is shim_wipfile
        assert "wipfile" in rt.__all__

    def test_open_and_inventory(self, wip):
        """Opening yields a populated inventory with the expected kinds present."""
        assert wip.filename == str(WIP_PATH)
        # Every list is (ID, caption) pairs.
        for entries in (wip.maps, wip.spectra, wip.images, wip.masks, wip.texts):
            assert all(isinstance(i, int) and isinstance(c, str) for i, c in entries)
        assert (FLG42_MAP_ID, "FLG_42--Scan Piezo--010--Spec.Data 1") in wip.maps
        assert any(i == FLG42_SPEC_ID for i, _ in wip.spectra)
        assert any(i == FLG42_MASK_ID for i, _ in wip.masks)
        assert any(i == FLG42_IMAGE_ID for i, _ in wip.images)
        # repr mentions the counts; list() prints without raising.
        assert "maps" in repr(wip)
        wip.list()
        wip.list(kinds=("map",))

    def test_lookup_by_caption(self, wip):
        """An exact caption resolves to the same object as its ID."""
        by_id = wip.ramanmap(FLG42_MAP_ID)
        by_caption = wip.ramanmap("FLG_42--Scan Piezo--010--Spec.Data 1")
        np.testing.assert_array_equal(by_id.mapxr.data, by_caption.mapxr.data)

    def test_lookup_errors(self, wip):
        """Unknown IDs / captions raise KeyError; wrong kind raises TypeError."""
        with pytest.raises(KeyError):
            wip.ramanmap(999999)
        with pytest.raises(KeyError):
            wip.ramanmap("no such caption")
        # A mask is not a map.
        with pytest.raises(TypeError):
            wip.ramanmap(FLG42_MASK_ID)
        # A map is not a spectrum.
        with pytest.raises(TypeError):
            wip.singlespec(FLG42_MAP_ID)
        # A map is not an image / mask either.
        with pytest.raises(TypeError):
            wip.mask(FLG42_MAP_ID)
        # Ambiguous caption: two "Fit Lorentz: Curves" maps share a caption.
        with pytest.raises(KeyError, match="ambiguous"):
            wip.ramanmap("FLG_42--Scan Piezo--010--Spec.Data 1 (Fit Lorentz: Curves)")

    def test_tree(self, wip, capsys):
        """tree() prints a per-scan ASCII tree with derived objects nested under the raw map.

        Checks the structure rather than the exact text: group headers are
        scan names, the raw map of the FLG_42 scan is a direct child, and
        objects derived from it (the extracted spectrum, the user-drawn
        mask, which has no caption prefix) sit one level deeper.
        """
        lines = wip._tree_lines()
        assert lines[0].startswith(str(WIP_PATH))
        # Top-level groups are drawn with the tree glyphs at column 0.
        groups = [l[4:] for l in lines if l.startswith(("├── ", "└── "))]
        assert "FLG_42--Scan Piezo--010" in groups
        assert groups[-1] == "(unlinked)"
        # Locate the FLG_42 group and its children.
        start = lines.index("├── FLG_42--Scan Piezo--010")
        block = []
        for l in lines[start + 1:]:
            if l.startswith(("├── ", "└── ")):
                break
            block.append(l)
        # Raw map is a direct child (depth 1); spectrum 200 and mask 254 are
        # nested beneath it (depth 2); the Information text is depth 1.
        depth1 = [l for l in block if l[4:8] in ("├── ", "└── ")]
        depth2 = [l for l in block if l[8:12] in ("├── ", "└── ")]
        assert any(f"[map]" in l and f" {FLG42_MAP_ID}  " in l for l in depth1)
        assert any("[text]" in l and "Information" in l for l in depth1)
        assert any("[spectrum]" in l and f" {FLG42_SPEC_ID}  " in l for l in depth2)
        assert any("[mask]" in l and f" {FLG42_MASK_ID}  " in l for l in depth2)
        # A raw caption with a bare suffix ("Spec.Data 1-anal0") still acts as
        # the raw node, so its "(Sub min)" copy nests under it.
        start = lines.index("├── Scan Piezo--004")
        sub = next(l for l in lines[start + 1:] if f" {FLG42_SMALL_MAP_ID}  " in l)
        assert sub[8:12] in ("├── ", "└── ")
        # SDK-internal objects only appear with show_all=True.
        assert not any("TDSpaceTransformation" in l for l in lines)
        assert any("TDSpaceTransformation" in l for l in wip._tree_lines(show_all=True))
        # tree() prints the same thing.
        wip.tree()
        assert capsys.readouterr().out.strip().splitlines() == lines

    def test_cli_inspect_by_id(self, capsys):
        """``ramantools inspect project.wip --id ID`` summarises one object.

        Without ``--id`` the whole tree is printed (covered by test_tree
        through the library call); with it the CLI must load the object
        and print its repr, or its summary dict with ``--json``.
        """
        from ramantools.cli import main  # local import: the CLI is not needed elsewhere here
        assert main(['inspect', str(WIP_PATH), '--id', '200']) == 0
        out = capsys.readouterr().out
        assert out.startswith('<ramantools.singlespec') and 'wip_extract' in out
        assert main(['inspect', str(WIP_PATH), '--id', '28', '--json']) == 0
        import json
        data = json.loads(capsys.readouterr().out)
        assert data['class'] == 'ramanmap' and data['pixel_x'] == 50 and data['pixel_y'] == 120

    def test_info_text(self, wip):
        """Information text is found for a map, for an extracted spectrum, and for a mask.

        The spectrum and the mask have no Information block of their own;
        the loader must fall back to the parent scan's text (by caption
        prefix for the spectrum, by shared space transformation for the mask).
        """
        for key in (FLG42_MAP_ID, FLG42_SPEC_ID, FLG42_MASK_ID):
            text = wip.info(key)
            assert text is not None
            assert text.splitlines()[0] == "FLG_42--Scan Piezo--010"
            assert "Excitation Wavelength [nm]:\t532.033" in text


class TestWipfileRamanmap:
    """ramanmap instances built from the project must match text-export ones."""

    def test_shape_dims_dtype(self, flg42_map):
        """mapxr has the standard (ramanshift, height, width) layout in float64."""
        assert isinstance(flg42_map, rt.ramanmap)
        assert flg42_map.mapxr.dims == ("ramanshift", "height", "width")
        assert flg42_map.mapxr.shape == FLG42_MAP_SHAPE
        # Raw WITec maps are uint16; the loader must upcast so that in-place
        # background subtraction does not truncate.
        assert flg42_map.mapxr.dtype == np.float64
        assert flg42_map.map.shape == FLG42_MAP_SHAPE
        assert flg42_map.ramanshift.shape == (FLG42_MAP_SHAPE[0],)
        # Raman shift axis is monotonic and in the expected window.
        assert np.all(np.diff(flg42_map.ramanshift) > 0)
        assert 1380 < flg42_map.ramanshift[0] < 1400
        assert 4600 < flg42_map.ramanshift[-1] < 4700

    def test_geometry_attributes(self, flg42_map):
        """Pixel counts and physical sizes come from the space transformation."""
        assert (flg42_map.pixel_x, flg42_map.pixel_y) == (FLG42_MAP_SHAPE[2], FLG42_MAP_SHAPE[1])
        assert (flg42_map.size_x, flg42_map.size_y) == pytest.approx(FLG42_MAP_SIZE)
        # Coordinates follow the same linspace convention as _toxarray.
        np.testing.assert_allclose(
            flg42_map.mapxr.width.data, np.linspace(0, FLG42_MAP_SIZE[0], FLG42_MAP_SHAPE[2])
        )
        np.testing.assert_allclose(
            flg42_map.mapxr.height.data, np.linspace(0, FLG42_MAP_SIZE[1], FLG42_MAP_SHAPE[1])
        )
        assert flg42_map.mapxr.attrs["map width"] == "32.0 um"
        assert flg42_map.mapxr.attrs["map height"] == "70.0 um"
        # Stage position is a finite number in µm (thousands on this stage).
        assert np.isfinite(flg42_map.positioner_x) and abs(flg42_map.positioner_x) > 100

    def test_instrument_metadata(self, flg42_map):
        """Metadata parsed from the embedded Information text lands in the usual attrs."""
        assert flg42_map.mapname == "FLG_42--Scan Piezo--010"
        assert flg42_map.laser == pytest.approx(FLG42_LASER_NM)
        assert flg42_map.itime == pytest.approx(FLG42_ITIME_S)
        assert flg42_map.samplename == FLG42_SAMPLENAME
        assert flg42_map.grating.startswith("G1")
        assert flg42_map.mapxr.attrs["laser excitation"] == f"{FLG42_LASER_NM} nm"
        assert flg42_map.mapxr.attrs["integration time"] == f"{FLG42_ITIME_S} s"
        # Info text was found → the magnification gets the conventional 'x' suffix.
        assert flg42_map.mapxr.attrs["objective magnification"].endswith("x")
        assert flg42_map.wipfilename == str(WIP_PATH)
        assert flg42_map.filename == str(WIP_PATH)
        # The raw text is kept for print_metadata().
        assert "Excitation Wavelength" in flg42_map.metadata
        flg42_map.print_metadata()

    def test_history_provenance(self, flg42_map):
        """The comments log starts like a file load and records the source object."""
        comments = flg42_map.mapxr.attrs["comments"]
        assert comments.startswith("raw data loaded")
        assert f"object ID {FLG42_MAP_ID}" in comments
        flg42_map.history()

    def test_orientation_matches_witec_image(self, wip):
        """Pixel ordering is right: the G-band sum of a map reproduces Witec's own G-Sum image.

        The project stores an image Witec computed from map 107 by summing
        the G band. Summing the same band in our extracted map must
        correlate ~1 with it; a transposed or vertically flipped layout
        gives ~0. This guards the [y][x] buffer interpretation.
        """
        m = wip.ramanmap(FLG42_SMALL_MAP_ID)
        gsum = wip.image(FLG42_GSUM_IMAGE_ID)
        assert gsum.shape == m.mapxr.shape[1:]
        band = m.mapxr.sel(ramanshift=slice(1500, 1660)).sum("ramanshift")
        corr = np.corrcoef(band.data.ravel(), gsum.data.ravel())[0, 1]
        assert corr > 0.99
        # Sanity: the transposed layout must NOT match.
        corr_t = np.corrcoef(band.data.T.ravel(), gsum.data.ravel())[0, 1]
        assert abs(corr_t) < 0.5

    def test_pipeline_runs(self, wip):
        """The standard processing chain works on a project-loaded map."""
        m = wip.ramanmap(FLG42_SMALL_MAP_ID)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_bg, coeff, covar = m.remove_bg()
            m_norm = m_bg.normalize(1582)
            m_crr = m_norm.crr()
        assert m_crr.mapxr.shape == m.mapxr.shape
        assert "background subtracted" in m_crr.mapxr.attrs["comments"]
        assert "normalized" in m_crr.mapxr.attrs["comments"]
        # Original is untouched (copy-on-modify).
        assert m.mapxr.attrs["units"] == "au"


class TestWipfileSinglespec:
    """singlespec instances built from extracted spectra in the project."""

    def test_spectrum_matches_parent_pixel(self, wip, flg42_map):
        """The "(31/33)" spectrum equals pixel x=31, y=33 of the parent map.

        Witec names extracted spectra by their zero-based (x/y) pixel. After
        the height flip in _toxarray, pixel row y sits at height index
        ``pixel_y - 1 - y``.
        """
        s = wip.singlespec(FLG42_SPEC_ID)
        assert isinstance(s, rt.singlespec)
        assert s.ssxr.dims == ("ramanshift",)
        assert s.ssxr.shape == (FLG42_MAP_SHAPE[0],)
        np.testing.assert_allclose(s.ramanshift, flg42_map.ramanshift)
        x, y = FLG42_SPEC_PIXEL
        np.testing.assert_allclose(
            s.counts, flg42_map.mapxr.data[:, FLG42_MAP_SHAPE[1] - 1 - y, x]
        )

    def test_spectrum_metadata(self, wip):
        """Instrument metadata is inherited from the parent scan; position is its own."""
        s = wip.singlespec(FLG42_SPEC_ID)
        assert s.specname.startswith("FLG_42--Scan Piezo--010")
        assert s.laser == pytest.approx(FLG42_LASER_NM)
        assert s.itime == pytest.approx(FLG42_ITIME_S)
        # Position from the spectrum's own space transformation (µm, finite).
        for v in (s.positioner_x, s.positioner_y, s.positioner_z):
            assert np.isfinite(v)
        assert s.ssxr.attrs["sample positioner Z"] == s.positioner_z
        assert f"object ID {FLG42_SPEC_ID}" in s.ssxr.attrs["comments"]

    def test_spectrum_pipeline(self, wip):
        """remove_bg → normalize works on a project-loaded spectrum."""
        s = wip.singlespec(FLG42_SPEC_ID)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_bg, _, _ = s.remove_bg()
            s_norm = s_bg.normalize(1582)
        assert s_norm.ssxr.attrs["units"] == " "


class TestWipfileImagesAndMasks:
    """Images and masks come out as 2-D DataArrays aligned with the map."""

    def test_mask(self, wip, flg42_map):
        """A Witec mask is a boolean DataArray on the map's (height, width) grid."""
        mask = wip.mask(FLG42_MASK_ID)
        assert isinstance(mask, xr.DataArray)
        assert mask.dims == ("height", "width")
        assert mask.dtype == bool
        assert mask.shape == flg42_map.mapxr.shape[1:]
        np.testing.assert_allclose(mask.width.data, flg42_map.mapxr.width.data)
        np.testing.assert_allclose(mask.height.data, flg42_map.mapxr.height.data)
        # The drawn region is non-empty but not the whole map.
        assert 0 < int(mask.sum()) < mask.size
        # Applying it to the map works directly via xarray alignment.
        masked = flg42_map.mapxr.where(mask)
        assert masked.shape == flg42_map.mapxr.shape
        assert int(np.isnan(masked.isel(ramanshift=0)).sum()) == int((~mask).sum())

    def test_image(self, wip, flg42_map):
        """A numeric image carries its Witec unit and the map's coordinates."""
        img = wip.image(FLG42_IMAGE_ID)
        assert isinstance(img, xr.DataArray)
        assert img.dims == ("height", "width")
        assert img.dtype == np.float64
        assert img.shape == flg42_map.mapxr.shape[1:]
        assert img.attrs["units"] == "rel. 1/cm"
        np.testing.assert_allclose(img.width.data, flg42_map.mapxr.width.data)

    def test_mask_from_image_thresholds(self, wip):
        """mask() accepts a numeric image and thresholds it at != 0."""
        m = wip.mask(FLG42_GSUM_IMAGE_ID)
        assert m.dtype == bool
        img = wip.image(FLG42_GSUM_IMAGE_ID)
        np.testing.assert_array_equal(m.data, img.data != 0)


class TestFromArraysConstructors:
    """The private array constructors behave sanely on their own."""

    def test_ramanmap_from_arrays_shape_check(self):
        """Mismatched array shape raises a clear ValueError, missing dims too."""
        rs = np.linspace(100, 3000, 10)
        good = np.zeros((10, 3, 4))
        fields = {"name": "t", "pixel_x": 4, "pixel_y": 3, "size_x": 2.0, "size_y": 1.5}
        m = rt.ramanmap._from_arrays(rs, good, filename="x.wip", fields=fields)
        assert m.mapxr.shape == (10, 3, 4)
        assert m.mapname == "t"
        # Sentinels fill the metadata that was not supplied.
        assert m.samplename == "N/A"
        with pytest.raises(ValueError, match="shape"):
            rt.ramanmap._from_arrays(rs, np.zeros((10, 4, 3)), filename="x.wip", fields=fields)
        with pytest.raises(ValueError, match="size_x"):
            rt.ramanmap._from_arrays(rs, good, filename="x.wip", fields={"pixel_x": 4, "pixel_y": 3})

    def test_singlespec_from_arrays(self):
        """Length mismatch raises; a good call yields a working singlespec."""
        rs = np.linspace(100, 3000, 10)
        s = rt.singlespec._from_arrays(rs, np.ones(10), filename="x.wip", fields={"name": "s"})
        assert s.ssxr.shape == (10,)
        assert s.specname == "s"
        assert np.isnan(s.positioner_z)
        with pytest.raises(ValueError, match="shape"):
            rt.singlespec._from_arrays(rs, np.ones(9), filename="x.wip", fields={})
