"""Witec data-file header parsing helpers.

Keeps the regex-heavy metadata extraction in one place, away from class
bodies. Moved out of ``ramantools/ramantools.py`` during the Phase 1
refactor for readability.
"""
import re  # Regular expressions for header parsing

def _parse_witec_datafile_header(header_text: str) -> dict:
        """Extract metadata from a Witec "table-export" data-file header.

        Handles both the Witec Project **v5** ("old table export") and **v7**
        ("new table export") header formats. Their ``[Header]`` blocks share
        the same set of key=value lines; the only observed difference is that
        v7 leaves ``FileName =`` empty while v5 fills in the source ``.wip``
        path. Maps include ``SizeX`` / ``SizeY`` (pixel counts),
        ``ScanWidth`` / ``ScanHeight`` (physical size in the unit given by
        ``ScanUnit``) and ``ScanOriginX/Y/Z``. Single spectra include
        ``PositionX/Y/Z`` and trivial ``SizeX = SizeY = 1``. This helper
        returns a dict of whatever fields it finds; missing fields are simply
        absent from the dict so callers can decide how to fall back.

        :param header_text: raw text of the header block as captured by
                :py:meth:`ramanmap._load_map` / :py:meth:`singlespec._load_singlespec`
                into the ``metadata_datafile`` attribute.
        :type header_text: str
        :return: dict keyed by attribute name (``pixel_x``, ``pixel_y``,
                ``size_x``, ``size_y``, ``positioner_x``, ``positioner_y``,
                ``positioner_z``, ``wipfilename``, ``graphname``)
        :rtype: dict
        """
        out: dict = {}
        if not header_text:
                # No header captured (bare data file). Nothing to extract.
                return out

        # Small helper that pulls the first regex match and casts it; returns
        # None (rather than raising) if the pattern is absent or the cast
        # fails — callers handle missing keys via dict.get / "in" checks.
        def _first(pattern: str, text: str, cast=str):
                matches = re.findall(pattern, text)
                if not matches:
                        return None
                try:
                        return cast(matches[0])
                except (ValueError, TypeError):
                        return None

        # Both map and spec headers carry FileName and GraphName.
        file_name = _first(r'(?<=FileName = ).+', header_text)
        if file_name is not None:
                # Strip in case the line ends with whitespace or \r.
                out['wipfilename'] = file_name.strip()
        graph_name = _first(r'(?<=GraphName = ).+', header_text)
        if graph_name is not None:
                out['graphname'] = graph_name.strip()

        # Integer pixel dimensions. Map files set these to real pixel counts;
        # spec files set them to 1. Either way they're correct to propagate.
        pix_x = _first(r'(?<=SizeX = )-?\d+', header_text, int)
        if pix_x is not None:
                out['pixel_x'] = pix_x
        pix_y = _first(r'(?<=SizeY = )-?\d+', header_text, int)
        if pix_y is not None:
                out['pixel_y'] = pix_y

        # Map-only: physical scan dimensions and scan origin. We don't guard
        # with a "map vs spec" flag — spec headers simply don't have these
        # keys, so the regex returns None and the dict keys are skipped.
        scan_width = _first(r'(?<=ScanWidth = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_width is not None:
                out['size_x'] = scan_width
        scan_height = _first(r'(?<=ScanHeight = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_height is not None:
                out['size_y'] = scan_height
        scan_origin_x = _first(r'(?<=ScanOriginX = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_origin_x is not None:
                out['positioner_x'] = scan_origin_x
        scan_origin_y = _first(r'(?<=ScanOriginY = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_origin_y is not None:
                out['positioner_y'] = scan_origin_y

        # Spec-only: absolute sample-positioner coordinates. Overwrites any
        # map-origin values if both appear (they shouldn't in practice).
        pos_x = _first(r'(?<=PositionX = )-?\d+(?:\.\d+)?', header_text, float)
        if pos_x is not None:
                out['positioner_x'] = pos_x
        pos_y = _first(r'(?<=PositionY = )-?\d+(?:\.\d+)?', header_text, float)
        if pos_y is not None:
                out['positioner_y'] = pos_y
        pos_z = _first(r'(?<=PositionZ = )-?\d+(?:\.\d+)?', header_text, float)
        if pos_z is not None:
                out['positioner_z'] = pos_z

        return out


def _graphname_to_name(graphname: str) -> str:
        """Normalize a Witec GraphName into a mapname/specname.

        GraphName lines in data headers look like
        ``MK_FLG_ABC_111--Spectrum--092--Spec.Data 1``; the info-file first line
        omits the trailing ``--Spec.Data N`` suffix. Strip it so attributes
        match between the info-file and header-only load paths.
        """
        # Remove the trailing "--Spec.Data <N>" token plus any trailing whitespace.
        return re.sub(r'--Spec\.Data\s*\d+\s*$', '', graphname).strip()


# ---------------------------------------------------------------------------
# Info-file ("metadata file") parsing
# ---------------------------------------------------------------------------

# Tuples of ``(attribute_name, regex, cast)`` describing the fields the
# per-class ``_load_info`` used to extract with inline regexes. Factoring
# the regexes into a table lets both classes share a single parser while
# still controlling which subset of fields they want.
#
# The regexes match the pre-refactor inline patterns verbatim — any drift
# here would show up as a pipeline-checksum mismatch via the safety-net
# tests, but we also rely on behavioural identity (e.g. `.[0]` raising
# IndexError for missing fields) to preserve the loud failure mode.
_INFO_FILE_COMMON = [
        ('date',          r'(?<=Start Date:\t)-?.+',                          str),
        ('time',          r'(?<=Start Time:\t)-?.+',                          str),
        # Sample Name uses ``.*`` rather than ``.+`` so an empty value
        # (trailing tab with no text) still matches — legacy behaviour.
        ('samplename',    r'(?<=Sample Name:\t).*',                           str),
        ('laser',         r'(?<=Excitation Wavelength \[nm\]:\t)-?.+',         float),
        ('itime',         r'(?<=Integration Time \[s\]:\t)-?.+',              float),
        ('grating',       r'(?<=Grating:\t)-?.+',                             str),
        # These two kept as raw (no r-prefix) because the original code did
        # the same; changing it could alter regex semantics in edge cases.
        ('objname',       '(?<=Objective Name:\t)-?.+',                       str),
        ('objmagn',       '(?<=Objective Magnification:\t)-?.+',              str),
        ('positioner_x',  r'(?<=Position X \[µm\]:\t)-?.+',                    float),
        ('positioner_y',  r'(?<=Position Y \[µm\]:\t)-?.+',                    float),
]

# Map-only fields: pixel counts + physical scan dimensions.
_INFO_FILE_MAP_ONLY = [
        ('pixel_x',  r'(?<=Points per Line:\t)-?\d+',          int),
        ('pixel_y',  r'(?<=Lines per Image:\t)-?\d+',          int),
        ('size_x',   r'(?<=Scan Width \[µm\]:\t)-?\d+\.\d+',    float),
        ('size_y',   r'(?<=Scan Height \[µm\]:\t)-?\d+\.\d+',   float),
]

# Spec-only field: the Z positioner coordinate that maps don't carry.
_INFO_FILE_SPEC_ONLY = [
        ('positioner_z', r'(?<=Position Z \[µm\]:\t)-?.+', float),
]


def _parse_info_file(info_path: str, *, is_map: bool):
        """Load and parse a Witec info (metadata) file.

        Returns ``(raw_text, fields)`` where ``fields`` is a dict keyed by
        the attribute the class will assign to — e.g. ``'laser'``,
        ``'pixel_x'``, ``'positioner_z'`` — plus a special ``'name'`` key
        holding the first line of the file (the map / spec name).

        Set ``is_map=True`` to include the pixel / scan-dimension fields;
        ``is_map=False`` swaps in ``positioner_z`` instead. Matches the
        pre-refactor inline regex bodies exactly so the pipeline-checksum
        safety net stays green.

        Missing required fields still raise ``IndexError`` — the same loud
        failure mode the original inline ``re.findall(...)[0]`` produced.
        """
        # latin1 encoding handles the µm symbol that appears in the
        # "Position X [µm]" / "Scan Width [µm]" header lines.
        with open(info_path, mode='r', encoding='latin1') as infofile:
                text = infofile.read()

        # First line is always the instance name (mapname / specname).
        # ``re.findall(r'.*', ...)[0]`` matches the original behaviour,
        # including the fact that it stops at the first newline.
        fields = {'name': re.findall(r'.*', text)[0]}
        table = _INFO_FILE_COMMON + (_INFO_FILE_MAP_ONLY if is_map else _INFO_FILE_SPEC_ONLY)
        for attr, pattern, cast in table:
                # ``[0]`` preserves the legacy "IndexError if the key is
                # missing" failure behaviour — no silent fallback to a
                # sentinel when the info file is present but malformed.
                fields[attr] = cast(re.findall(pattern, text)[0])
        return text, fields
