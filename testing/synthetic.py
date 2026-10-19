"""Synthetic Witec-format Raman data for tests and minimal examples.

The real fixtures under ``test-data/`` are a 100 MB map and a 260 MB
WITec project that are not in the repository, so a fresh clone cannot
run most of the test suite. This module generates small files in the
same text-export layout the Witec software produces (``[Header]`` block
plus tab-separated table, latin-1 encoded, ``µm`` in the info file) so
that loading, background subtraction, calibration, normalisation, cosmic
ray removal, peak masking, peak fitting and the save / load round trip
can all be exercised anywhere.

The spectra are modelled on the measured graphene-on-silicon data in
``test-data/``: a flat CCD baseline near 880 counts, the silicon line
at 520 cm⁻¹ (FWHM ≈ 19), the graphene G band at 1582 cm⁻¹ (FWHM ≈ 22)
and the 2D band at 2700 cm⁻¹ (FWHM ≈ 60), the same 1600-point
-640 … 4020 cm⁻¹ axis, low read noise, and the occasional one-pixel
cosmic-ray spike. The map is a flake: an ellipse where the graphene
bands exist and the silicon line is attenuated, on a bare-substrate
background, with a mild baseline gradient across the scan so that
per-pixel background subtraction has something to do.

Every generator takes a ``seed`` and is fully deterministic. The
``write_*`` functions return the ground truth (peak positions,
amplitudes, cosmic-ray positions, flake mask) alongside the file paths
so tests can assert against known values.

Run as a script to drop a sample set into a directory::

    python testing/synthetic.py /tmp/raman-synthetic
"""
from __future__ import annotations

import sys  # command-line entry point
from pathlib import Path  # output paths
import numpy as np  # type: ignore

# ---------------------------------------------------------------------------
# Spectral model
# ---------------------------------------------------------------------------

# Peak positions (1/cm) and shapes taken from the measured test data.
# ``ampl`` is the peak height above the baseline, ``width`` the FWHM.
SI_PEAK = {'x0': 520.5, 'ampl': 1200.0, 'width': 19.0}
G_PEAK = {'x0': 1582.0, 'ampl': 1150.0, 'width': 22.0}
TWO_D_PEAK = {'x0': 2700.0, 'ampl': 450.0, 'width': 60.0}

# Raman shift axis: same span and length as the Witec exports.
SHIFT_START = -640.0
SHIFT_STOP = 4020.0
N_POINTS = 1600

# CCD baseline (counts) and the gentle slope seen in the real data.
BASELINE = 880.0
BASELINE_SLOPE = 0.004  # counts per 1/cm
NOISE_SIGMA = 2.0  # read noise, counts
COSMIC_HEIGHT = 2500.0  # one-pixel spike height, counts

# Metadata written to the info files / headers. Chosen to look like the
# instrument that produced the real fixtures.
LASER_NM = 487.986
GRATING = 'G1: 600 g/mm BLZ=500nm'
OBJECTIVE = 'Zeiss EC Epiplan-Neofluar Dic 50x / 0.8'
MAGNIFICATION = '50.0'


def lorentz(x, x0: float, ampl: float, width: float):
        """Lorentzian with height ``ampl`` and FWHM ``width`` (no offset).

        Same parametrisation as :func:`ramantools.lorentz`, so a fit with
        the default shape function should return these numbers.
        """
        return ampl * (width / 2) ** 2 / ((x - x0) ** 2 + (width / 2) ** 2)


def ramanshift_axis(n_points: int = N_POINTS) -> np.ndarray:
        """The Raman shift axis (1/cm), evenly spaced like the exports."""
        return np.linspace(SHIFT_START, SHIFT_STOP, n_points)


def model_spectrum(shift: np.ndarray, *, flake: float = 1.0, baseline: float = BASELINE,
                   miscalibration: float = 0.0) -> np.ndarray:
        """Noise-free spectrum for one pixel.

        :param flake: 0 on bare substrate (silicon line only), 1 on the
                flake (graphene bands present, silicon attenuated); values
                in between interpolate, which gives the map a soft edge.
        :param baseline: CCD offset in counts for this pixel.
        :param miscalibration: shift added to *all* peak positions (1/cm),
                i.e. the spectrometer error ``calibrate`` should recover.
        """
        y = baseline + BASELINE_SLOPE * (shift - SHIFT_START)
        # Silicon line is partly screened by the flake.
        y += lorentz(shift, SI_PEAK['x0'] + miscalibration, SI_PEAK['ampl'] * (1 - 0.3 * flake), SI_PEAK['width'])
        y += flake * lorentz(shift, G_PEAK['x0'] + miscalibration, G_PEAK['ampl'], G_PEAK['width'])
        y += flake * lorentz(shift, TWO_D_PEAK['x0'] + miscalibration, TWO_D_PEAK['ampl'], TWO_D_PEAK['width'])
        return y


def make_spectrum(*, seed: int = 0, flake: float = 1.0, miscalibration: float = 0.0,
                  cosmic_rays: int = 1, n_points: int = N_POINTS):
        """Synthetic single spectrum.

        Returns ``(ramanshift, counts, truth)`` where ``truth`` lists the
        cosmic-ray sample indices and the peak parameters actually used.
        Cosmic rays are placed away from the peaks (above 3000 cm⁻¹) so
        they never distort a fit.
        """
        rng = np.random.default_rng(seed)
        shift = ramanshift_axis(n_points)
        counts = model_spectrum(shift, flake=flake, miscalibration=miscalibration)
        counts += rng.normal(0.0, NOISE_SIGMA, size=shift.shape)
        # Cosmic rays: isolated single-sample spikes in the featureless
        # high-shift region.
        candidates = np.flatnonzero(shift > 3000)
        cr_idx = np.sort(rng.choice(candidates, size=cosmic_rays, replace=False)) if cosmic_rays else np.array([], int)
        counts[cr_idx] += COSMIC_HEIGHT
        # Witec exports integer CCD counts.
        counts = np.round(counts)
        truth = {
                'cosmic_ray_indices': cr_idx.tolist(),
                'miscalibration': miscalibration,
                'peaks': {
                        'si': dict(SI_PEAK, x0=SI_PEAK['x0'] + miscalibration, ampl=SI_PEAK['ampl'] * (1 - 0.3 * flake)),
                        'g': dict(G_PEAK, x0=G_PEAK['x0'] + miscalibration, ampl=G_PEAK['ampl'] * flake),
                        '2d': dict(TWO_D_PEAK, x0=TWO_D_PEAK['x0'] + miscalibration, ampl=TWO_D_PEAK['ampl'] * flake),
                },
        }
        return shift, counts, truth


def flake_mask(pixel_x: int, pixel_y: int) -> np.ndarray:
        """Soft elliptical flake, shape ``(pixel_y, pixel_x)``, values in 0 … 1.

        The ellipse is centred slightly off the middle so the centre pixel
        (which ``remove_bg`` / ``normalize`` use as reference by default)
        is well inside the flake while the edges are bare substrate.
        """
        yy, xx = np.mgrid[0:pixel_y, 0:pixel_x]
        cx, cy = 0.5 * (pixel_x - 1), 0.5 * (pixel_y - 1)
        rx, ry = 0.36 * pixel_x, 0.36 * pixel_y
        r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        # 1 inside r<0.85, 0 outside r>1.15, linear ramp between.
        return np.clip((1.15 - r) / 0.3, 0.0, 1.0)


def make_map(*, pixel_x: int = 12, pixel_y: int = 10, seed: int = 0, miscalibration: float = 0.0,
             cosmic_rays: int = 3, n_points: int = N_POINTS):
        """Synthetic map cube shaped ``(n_points, pixel_y, pixel_x)``.

        Row 0 of the cube is the *top* row of the scan, the convention of
        the Witec table export (``ramanmap._toxarray`` flips it). Returns
        ``(ramanshift, cube, truth)``; ``truth['flake']`` is the soft mask
        and ``truth['cosmic_rays']`` the ``(row, col, index)`` triples.
        """
        rng = np.random.default_rng(seed)
        shift = ramanshift_axis(n_points)
        flake = flake_mask(pixel_x, pixel_y)
        # Baseline drifts across the scan (left → right) by ~5 %.
        baseline = BASELINE * (1 + 0.05 * np.linspace(0, 1, pixel_x))[None, :] * np.ones((pixel_y, 1))
        cube = np.empty((n_points, pixel_y, pixel_x))
        for j in range(pixel_y):
                for i in range(pixel_x):
                        cube[:, j, i] = model_spectrum(shift, flake=float(flake[j, i]), baseline=float(baseline[j, i]),
                                                       miscalibration=miscalibration)
        cube += rng.normal(0.0, NOISE_SIGMA, size=cube.shape)
        candidates = np.flatnonzero(shift > 3000)
        crs = []
        for _ in range(cosmic_rays):
                j, i = int(rng.integers(pixel_y)), int(rng.integers(pixel_x))
                k = int(rng.choice(candidates))
                cube[k, j, i] += COSMIC_HEIGHT
                crs.append((j, i, k))
        cube = np.round(cube)
        truth = {'flake': flake, 'cosmic_rays': crs, 'miscalibration': miscalibration, 'baseline': baseline}
        return shift, cube, truth


# ---------------------------------------------------------------------------
# Witec text-export writers
# ---------------------------------------------------------------------------

# latin-1 like the real exports; ``µ`` is the one non-ASCII character.
ENCODING = 'latin1'


def _fmt_row(values) -> str:
        """One table row in Witec's ``%.5E`` notation, tab separated."""
        return '\t'.join(f'{v: .5E}' for v in values)


def _info_common(name: str, sample: str, itime: float) -> str:
        """Header of an info file shared by maps and spectra (tabs matter)."""
        return (
                f'{name}\n\n'
                'General:\n'
                'System ID:\t100-1100-758\n'
                'Start Time:\t12:00:00\n'
                'Start Date:\t2026. október 19., Monday\n'
                'Duration:\t0h 1m 0s\n'
                'User Name:\tWitec\n'
                f'Sample Name:\t{sample}\n'
                'Configuration:\tRaman CCD1\n\n'
        )


def _info_instrument(itime: float) -> str:
        """Spectrometer / camera / objective blocks of an info file."""
        return (
                'UHTS300M:\n'
                f'Excitation Wavelength [nm]:\t{LASER_NM}\n'
                f'Grating:\t{GRATING}\n'
                'Center Wavelength [nm]:\t540.752\n'
                'Spectral Center [rel. 1/cm]:\t1999.637\n\n'
                'DU970_BV:\n'
                'Width [Pixels]:\t1600\n'
                'Height [Pixels]:\t200\n'
                'Temperature [°C]:\t-59\n\n'
                f'Integration Time [s]:\t{itime}\n\n'
                'Objective:\n'
                f'Objective Name:\t{OBJECTIVE}\n'
                f'Objective Magnification:\t{MAGNIFICATION}\n\n'
        )


def write_spectrum(directory, *, name: str = 'SYN_FLAKE--Spectrum--001', sample: str = 'SYN_FLAKE',
                   header: bool = True, position=(100.0, -200.0, 0.0), itime: float = 3.0, **kwargs) -> dict:
        """Write a synthetic single spectrum + info file into ``directory``.

        ``kwargs`` go to :func:`make_spectrum`. With ``header=False`` the
        data file is a bare two-column table (the Witec Project v5 style
        without ``[Header]``) so the info-file-only load path is covered.
        Returns a dict with ``spec_path``, ``info_path``, the arrays and
        the ``truth`` from the generator.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shift, counts, truth = make_spectrum(**kwargs)
        spec_path = directory / f'{name}.txt'
        info_path = directory / f'{name}_info.txt'
        with open(spec_path, 'w', encoding=ENCODING, newline='\n') as fh:
                if header:
                        fh.write(
                                '//Exported ASCII-File\n[Header]\n'
                                'FileName = \n'
                                f'GraphName = {name}--Spec.Data 1\n'
                                'SizeX = 1\nSizeY = 1\n'
                                f'SizeGraph = {len(shift)}\n'
                                'PositionUnit = µm\n'
                                f'PositionX = {position[0]}\nPositionY = {position[1]}\nPositionZ = {position[2]}\n'
                                'XAxisUnit = rel. 1/cm\nDataUnit = CCD cts\n\n[Data]\n'
                                f'X-Axis\t{name}--Spec.Data 1\nrel. 1/cm\tCCD cts\n'
                        )
                for x, y in zip(shift, counts):
                        fh.write(_fmt_row((x, y)) + '\n')
        with open(info_path, 'w', encoding=ENCODING, newline='\n') as fh:
                fh.write(_info_common(name, sample, itime))
                fh.write(_info_instrument(itime))
                fh.write(
                        'Sample Location (global position):\n'
                        f'Position X [µm]:\t{position[0]:.3f}\n'
                        f'Position Y [µm]:\t{position[1]:.3f}\n'
                        f'Position Z [µm]:\t{position[2]:.3f}\n'
                )
        return {'spec_path': spec_path, 'info_path': info_path, 'ramanshift': shift, 'counts': counts,
                'truth': truth, 'name': name, 'sample': sample}


def write_map(directory, *, name: str = 'SYN_FLAKE--Scan Piezo--001', sample: str = 'SYN_FLAKE',
              header: bool = True, size_x: float = 12.0, size_y: float = 10.0, origin=(100.0, -200.0),
              itime: float = 1.0, **kwargs) -> dict:
        """Write a synthetic map + info file into ``directory``.

        ``kwargs`` go to :func:`make_map` (``pixel_x``, ``pixel_y``, ``seed``,
        ``miscalibration``, ``cosmic_rays``). The table has the ramanshift
        in column 0 followed by one column per pixel in row-major order,
        exactly like a Witec "table" export. Returns a dict with
        ``map_path``, ``info_path``, the cube and the ``truth``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shift, cube, truth = make_map(**kwargs)
        n_points, pixel_y, pixel_x = cube.shape
        map_path = directory / f'{name}.txt'
        info_path = directory / f'{name}_info.txt'
        with open(map_path, 'w', encoding=ENCODING, newline='\n') as fh:
                if header:
                        fh.write(
                                '//Exported ASCII-File\n[Header]\n'
                                'FileName = \n'
                                f'GraphName = {name}--Spec.Data 1\n'
                                f'SizeX = {pixel_x}\nSizeY = {pixel_y}\n'
                                f'SizeGraph = {n_points}\n'
                                'ScanUnit = µm\n'
                                f'ScanWidth = {size_x}\nScanHeight = {size_y}\n'
                                f'ScanOriginX = {origin[0]}\nScanOriginY = {origin[1]}\nScanOriginZ = 0\n'
                                'XAxisUnit = rel. 1/cm\nDataUnit = CCD cts\n\n[Data]\n'
                        )
                flat = cube.reshape(n_points, pixel_y * pixel_x)
                for k in range(n_points):
                        fh.write(_fmt_row(np.concatenate(([shift[k]], flat[k]))) + '\n')
        with open(info_path, 'w', encoding=ENCODING, newline='\n') as fh:
                fh.write(_info_common(name, sample, itime))
                fh.write(
                        'Image Scan:\n'
                        f'Points per Line:\t{pixel_x}\n'
                        f'Lines per Image:\t{pixel_y}\n'
                        f'Scan Width [µm]:\t{size_x:.3f}\n'
                        f'Scan Height [µm]:\t{size_y:.3f}\n'
                        f'Scan Origin X [µm]:\t{origin[0]:.3f}\n'
                        f'Scan Origin Y [µm]:\t{origin[1]:.3f}\n'
                        'Scan Origin Z [µm]:\t0.000\n'
                        'Integration Time [s]:\t' + f'{itime}\n\n'
                )
                fh.write(_info_instrument(itime))
                fh.write(
                        'Sample Positioner:\n'
                        f'Position X [µm]:\t{origin[0]:.3f}\n'
                        f'Position Y [µm]:\t{origin[1]:.3f}\n'
                )
        return {'map_path': map_path, 'info_path': info_path, 'ramanshift': shift, 'cube': cube,
                'truth': truth, 'name': name, 'sample': sample, 'size_x': size_x, 'size_y': size_y}


def write_sample_set(directory) -> dict:
        """Write one map and one spectrum with a 2.5 cm⁻¹ miscalibration.

        This is the set the test suite uses and what the script entry
        point produces; the miscalibration gives ``calibrate`` a known
        answer to recover.
        """
        spec = write_spectrum(directory, miscalibration=2.5, seed=1)
        mp = write_map(directory, miscalibration=2.5, seed=2)
        return {'spectrum': spec, 'map': mp}


if __name__ == '__main__':  # pragma: no cover
        target = sys.argv[1] if len(sys.argv) > 1 else 'synthetic-data'
        files = write_sample_set(target)
        for group in files.values():
                for key in ('spec_path', 'map_path', 'info_path'):
                        if key in group:
                                print(group[key])
