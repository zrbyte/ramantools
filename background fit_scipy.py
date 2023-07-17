
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def linear_model(x, m, c):
    return m * x + c

def analyze_data(data, min_height, min_width, exclusion_factor, peak_positions=None):
    if peak_positions is None:
        # Find the peaks with specified minimum height and width
        peak_properties = find_peaks(data[:, 1], height=min_height, width=min_width, rel_height=0.5)

        # Find the indices of the peaks
        peak_indices = peak_properties[0]

        # Get the properties of the peaks
        peak_info = peak_properties[1]
    else:
        # Use the provided peak positions
        peak_indices = []
        for peak_position in peak_positions:
            # Find the index of the closest data point to the peak position
            closest_index = np.argmin(np.abs(data[:, 0] - peak_position))
            peak_indices.append(closest_index)

        # Calculate the widths of the peaks using the data
        peak_widths = [min_width] * len(peak_indices)  # Use the minimum width if peak widths cannot be calculated from the data
        peak_info = {'widths': peak_widths}

    # Calculate the start and end indices of each peak with a larger exclusion area
    start_indices = peak_indices - (exclusion_factor * np.array(peak_info['widths'])).astype(int)
    end_indices = peak_indices + (exclusion_factor * np.array(peak_info['widths'])).astype(int)

    # Ensure indices are within data bounds
    start_indices = np.maximum(start_indices, 0)
    end_indices = np.minimum(end_indices, len(data) - 1)

    # Define the indices covered by the peaks
    covered_indices = []
    for start, end in zip(start_indices, end_indices):
        covered_indices.extend(range(start, end + 1))

    # Remove these indices from the data
    mask = np.ones(data.shape[0], dtype=bool)
    mask[covered_indices] = False
    uncovered_data = data[mask]

    # Fit a line to the remaining data using scipy's curve_fit
    popt, _ = curve_fit(linear_model, uncovered_data[:, 0], uncovered_data[:, 1])

    # Get the line parameters
    slope, intercept = popt

    # Calculate the fitted line values
    line_values = linear_model(data[:, 0], slope, intercept)

    # Create an array for the line subtracted data
    subtracted_data = data.copy()
    subtracted_data[:, 1] -= line_values

    # Plot the data
    plt.figure(figsize=(10,6))
    plt.plot(data[:, 0], data[:, 1], label='Data')

    # Highlight the peaks
    plt.scatter(data[peak_indices, 0], data[peak_indices, 1], color='red', label='Peaks')

    # Plot the fitted line
    plt.plot(data[:, 0], line_values, color='blue', label='Fitted Line')

    # Highlight the new background data used for fitting
    plt.scatter(uncovered_data[:, 0], uncovered_data[:, 1], color='red', lw=3, alpha=0.5, label='Background')

    plt.xlabel('Column 1')
    plt.ylabel('Column 2')
    plt.title('Data Plot with Peaks, Fitted Line and Background Highlighted')
    plt.legend()
    plt.show()

    return {'slope': slope, 'intercept': intercept}, subtracted_data
    