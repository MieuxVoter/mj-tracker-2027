import matplotlib.colors as mcolors

_GRADE_COLORS_HEX = ["#990000", "#C23D13", "#C27C13", "#C2B113", "#D3D715", "#A0CF1C", "#3A9918"]


def get_grade_color_palette(n_colors: int, return_type: str = "rgb_tuples"):
    """
    Generates a list of RGB color tuples or a Matplotlib ListedColormap
    based on a specified number of grades.

    The color selection logic picks a centered subset from a predefined
    list of 7 base colors, mirroring the effective palette selection approach
    derived from the JavaScript 'getGradeColor' function for a given number of grades.

    Args:
        n_colors (int): The desired number of colors in the palette.
                          Must be an integer between 1 and
                          len(_GRADE_COLORS_HEX) (which is 7), inclusive.
        return_type (str): Specifies the desired output format.
                           - "rgb_tuples" (default): Returns a list of RGB tuples,
                             where each tuple is (R, G, B) with values from 0-255.
                           - "colormap": Returns a matplotlib.colors.ListedColormap
                             object.

    Returns:
        list[tuple[int, int, int]] or matplotlib.colors.ListedColormap:
            The generated color palette in the specified format.

    Raises:
        ValueError: If num_grades is out of the valid range (1 to
                    len(_GRADE_COLORS_HEX)) or if return_type is invalid,
                    or if a hex color string is malformed.
    """

    # Helper function to convert hex to RGB (0-255)
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        h = hex_color.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"Invalid hex color format: '{hex_color}'. Expected 6 hex digits.")
        try:
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            raise ValueError(f"Invalid character in hex color: '{hex_color}'")

    max_colors = len(_GRADE_COLORS_HEX)
    if not (1 <= n_colors <= max_colors):
        raise ValueError(f"Number of grades must be between 1 and {max_colors}, inclusive. " f"Received {num_grades}.")

    # Calculate the slice of colors to use, centered.
    extra_colors = max_colors - n_colors
    start_index = extra_colors // 2  # Integer division for Math.floor

    # selected_hex_colors will have exactly 'num_grades' elements
    selected_hex_colors = _GRADE_COLORS_HEX[start_index : start_index + n_colors]

    rgb_tuples_255 = [_hex_to_rgb(hc) for hc in selected_hex_colors]

    if return_type == "rgb_tuples":
        return rgb_tuples_255
    elif return_type == "colormap":
        # Normalize RGB tuples from 0-255 to 0.0-1.0 for ListedColormap
        rgb_tuples_normalized = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in rgb_tuples_255]
        return mcolors.ListedColormap(rgb_tuples_normalized, name=f"GradePalette_{num_grades}")
    else:
        raise ValueError(f"Invalid return_type: '{return_type}'. Choose 'rgb_tuples' or 'colormap'.")


# --- Example Usage ---
if __name__ == "__main__":
    print("Requesting palette for 5 grades (RGB tuples):")
    palette_5_rgb = get_grade_color_palette(n_colors=5, return_type="rgb_tuples")
    for i, rgb in enumerate(palette_5_rgb):
        # For 5 grades, start_index = (7-5)//2 = 1. Slice is [1:1+5] = [1:6]
        # Corresponds to _GRADE_COLORS_HEX[1] to _GRADE_COLORS_HEX[5]
        print(f"  {_GRADE_COLORS_HEX[1 + i]} -> {rgb}")
    # Expected HEX: #C23D13, #C27C13, #C2B113, #D3D715, #A0CF1C

    print("\nRequesting palette for 3 grades (Matplotlib Colormap):")
    palette_3_cmap = get_grade_color_palette(n_colors=3, return_type="colormap")
    print(f"  Colormap object: {palette_3_cmap}")
    print(f"  Colors in colormap (normalized 0-1):")
    # For 3 grades, start_index = (7-3)//2 = 2. Slice is [2:2+3] = [2:5]
    # Corresponds to _GRADE_COLORS_HEX[2] to _GRADE_COLORS_HEX[4]
    for i, color_norm in enumerate(palette_3_cmap.colors):
        original_hex = _GRADE_COLORS_HEX[2 + i]
        print(f"    {original_hex} -> ({color_norm[0]:.3f}, {color_norm[1]:.3f}, {color_norm[2]:.3f})")
    # Expected HEX: #C27C13, #C2B113, #D3D715

    print("\nRequesting palette for all 7 grades (default return_type 'rgb_tuples'):")
    palette_7_rgb = get_grade_color_palette(n_colors=7)
    for i, rgb in enumerate(palette_7_rgb):
        print(f"  {_GRADE_COLORS_HEX[i]} -> {rgb}")

    print("\n--- Error Handling Examples ---")
    try:
        print("Trying to request too many grades (e.g., 8):")
        get_grade_color_palette(n_colors=8)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    try:
        print("\nTrying to request zero grades:")
        get_grade_color_palette(n_colors=0)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    try:
        print("\nTrying an invalid return type:")
        get_grade_color_palette(n_colors=3, return_type="hex_list")
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    # Example of using the colormap with Matplotlib (requires matplotlib.pyplot)
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # if palette_3_cmap:
    #     # Create dummy grade names for the colorbar ticks
    #     gradeNames = [f"Grade {i+1}" for i in range(palette_3_cmap.N)]
    #
    #     data = np.random.rand(10, 10) * (palette_3_cmap.N -1) # Example data
    #     plt.figure(figsize=(6,4))
    #     plt.imshow(data, cmap=palette_3_cmap)
    #
    #     # Add a colorbar with grade names
    #     cbar = plt.colorbar(ticks=np.arange(palette_3_cmap.N))
    #     if len(gradeNames) == palette_3_cmap.N:
    #         cbar.ax.set_yticklabels(gradeNames)
    #
    #     plt.title(f"Plot with {palette_3_cmap.N}-Grade Custom Colormap")
    #     plt.show()
