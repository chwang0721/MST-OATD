import argparse

def float_or_int_tuple(value):
    """Custom type function to accept either float or tuple of ints."""
    try:
        # Case 1: Try parsing as a float
        return float(value)
    except ValueError:
        try:
            # Case 2: Try parsing as a tuple of ints (e.g., "1,2")
            # Split by commas and convert to integers
            parts = [int(x.strip()) for x in value.split(',')]
            return tuple(parts)
        except (ValueError, AttributeError):
            raise argparse.ArgumentTypeError(
                f"Must be a float or tuple of ints (e.g., '3.14' or '2,3'), got '{value}'"
            )

parser = argparse.ArgumentParser()
parser.add_argument("--grid_size", type=float_or_int_tuple, 
                    help="Input value (float or tuple of ints, standart values for porto e.g., '0.1' or '167,154')", default=0.1)
parser.add_argument("--dataset", type=str, default='porto')
parser.add_argument("--max_traj_time_delta", type=int, default=1900)
parser.add_argument("--find_boundary", type=bool, default=False)

parser.add_argument("--processes", type=int, default=10)
parser.add_argument("--chunk_size", type=int, default=8000)

args = parser.parse_args()