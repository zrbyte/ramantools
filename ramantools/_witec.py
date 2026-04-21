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
