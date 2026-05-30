import os
import re
import numpy as np
import tifffile


def run_concat_tif(input_dir, output_dir, status_callback=None):
    """
    Groups and merges 2D Z-slice TIFF files from input_dir into 3D stacked TIFF files
    based on Row, Col, Field, and Channel, then saves the results to output_dir.

    :param input_dir: Input directory path (containing 2D slices)
    :param output_dir: Output directory path (saving 3D stacked results)
    :param status_callback: Callback function for real-time progress transmission to the GUI, signature: def callback(msg: str)
    """

    def log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    if not os.path.exists(input_dir):
        log(f"Error: Input path does not exist -> {input_dir}")
        return False, "Input path does not exist"

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            log(f"Creating output directory: {output_dir}")
        except Exception as e:
            log(f"Error: Unable to create output directory -> {str(e)}")
            return False, f"Failed to create output directory: {str(e)}"

    # Regex matching: r(two digits)c(two digits)f(two digits)p(two digits)-ch(one digit)
    pattern = r'r(\d{2})c(\d{2})f(\d{2})p(\d{2})-ch(\d)\w*\.tiff?'
    p = re.compile(pattern)

    # Scan directory
    try:
        file_name_list = os.listdir(input_dir)
    except Exception as e:
        log(f"Error: Failed to read directory -> {str(e)}")
        return False, "Failed to read directory"

    # Efficient grouping using dictionary
    # Structure: { (row, col, field, channel): [(z_plane_idx, file_path), ...] }
    groups = {}

    log("Parsing filenames and grouping...")
    for name in file_name_list:
        match = p.search(name)
        if not match:
            continue  # Automatically ignore non-tiff files (e.g. xml description files) that do not match the pattern

        row, col, field, p_plane, channel = match.groups()
        group_key = (row, col, field, channel)
        z_plane = int(p_plane)  # Convert to integer to prevent character sorting errors
        file_path = os.path.join(input_dir, name)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append((z_plane, file_path))

    total_groups = len(groups)
    if total_groups == 0:
        log("No TIFF files matching the naming convention (e.g., r01c01f01p01-ch1.tiff) were found.")
        return False, "No matching TIFF files found"

    log(f"Parsing completed. Found {total_groups} group(s) of 3D data to be merged.")

    success_count = 0
    # Process each group for 3D stacking
    for idx, (group_key, file_tuples) in enumerate(groups.items()):
        # 1. Strictly sort by Z-plane ascending to ensure proper 3D spatial order
        file_tuples.sort(key=lambda x: x[0])
        sorted_filepaths = [item[1] for item in file_tuples]

        row, col, field, channel = group_key
        output_name = f'r{row}c{col}f{field}-ch{channel}.tiff'
        output_path = os.path.join(output_dir, output_name)

        log(f"[{idx + 1}/{total_groups}] Merging: {output_name} (containing {len(sorted_filepaths)} slice(s))...")

        try:
            # 2. Read all 2D slices in the group and perform 3D stacking
            stacked = np.stack([tifffile.imread(f) for f in sorted_filepaths])

            # 3. Write out the 3D TIFF file
            tifffile.imwrite(output_path, stacked)
            success_count += 1
        except Exception as e:
            log(f"Failed to process group {group_key}: {str(e)}")

    final_msg = f"Processing completed! Successfully merged {success_count}/{total_groups} 3D TIFF file(s)."
    log(final_msg)
    return True, final_msg