"""Shared Witec text-file loading for ramanmap and singlespec.

The pre-refactor ``_load_map`` and ``_load_singlespec`` methods were
near-identical: both scanned the top of the file for the first numeric
data line (to detect the variable-length ``[Header]`` block), captured
the header as ``metadata_datafile``, extracted the WIP filename, and
then handed the numeric table to ``np.loadtxt``. They differed only in
how the loaded matrix was reshaped at the end (2-D grid vs 2-column
table).

This module owns the shared scan + load path. Each class's ``_load_*``
method now boils down to one call plus class-specific post-processing.
"""
import re  # For header detection and FileName extraction
import numpy as np  # type: ignore


def _load_witec_datafile(path: str, *, scan_lines: int = 40):
        """Scan a Witec export file for its header, then numpy-load the data.

        Returns ``(toskip, metadata_datafile, wipfilename, matrix)`` where:

        * ``toskip`` — number of header lines skipped. 0 means the file is
          a bare "data-only" table with no ``[Header]`` block.
        * ``metadata_datafile`` — the captured header text, joined by
          ``'\\n'``. Empty string when there is no header.
        * ``wipfilename`` — value of the ``FileName =`` line in the header
          (or the data path itself when the file has no header). Empty
          string if the ``FileName =`` line appears but is empty (new v7
          table exports do this).
        * ``matrix`` — the result of ``np.loadtxt(path, skiprows=toskip,
          encoding='latin1')``. Callers reshape this to their class shape
          (1-D spectrum → first col + second col; 2-D map → ramanshift +
          reshape of remaining cols).

        ``scan_lines=40`` matches the legacy scan depth; the header is
        always far shorter in practice but the loose upper bound forgives
        malformed exports.
        """
        # First pass: read a bounded window from the top of the file so we
        # can locate the first data row without loading the whole file.
        with open(path, 'r', encoding='latin1') as file:
                lines = []
                for _ in range(scan_lines):
                        line = file.readline()
                        if not line:  # end of file reached before scan_lines
                                break
                        lines.append(line.strip())

        # Detect the first data row. Pattern: one or more floating-point
        # tokens on the line. Same regex the original ``_load_map`` used.
        data_pattern = r'(\d+\.\d+.+[\t ]+)+'
        toskip = 0
        for idx, line in enumerate(lines, start=0):
                if re.search(data_pattern, line):
                        toskip = idx
                        break  # Stop at the first match — that's our data start.

        # Capture the header block + wipfilename.
        if toskip == 0:
                # No header — treat the file path itself as the "wip filename"
                # so downstream code always has something to show.
                metadata_datafile = ''
                wipfilename = path
        else:
                # Re-read the header lines from scratch — using the cached
                # stripped copy is not safe because the tight loop above
                # may have truncated short files.
                with open(path, 'r', encoding='latin1') as file:
                        header_lines = [next(file).strip() for _ in range(toskip)]
                metadata_datafile = '\n'.join(header_lines)
                # ``[ \t]*`` (NOT ``\s*``) tolerates stripped trailing
                # whitespace: newer v7 exports emit ``FileName = \n`` that
                # ``strip()`` reduces to ``FileName =``. The ``if matches``
                # guard defends against the FileName line being absent.
                matches = re.findall(r'FileName =[ \t]*(.*?)(?:\n|$)', metadata_datafile)
                wipfilename = matches[0] if matches else ''

        # Second pass: numpy-load the numeric table. This reopens the file
        # from scratch so the scan above doesn't interfere with offsets.
        matrix = np.loadtxt(path, skiprows=toskip, encoding='latin1')
        return toskip, metadata_datafile, wipfilename, matrix
