# check_errors.py
import sys
from nbformat import read

def check_error_tags(nbfile) -> bool:
    with open(nbfile) as f:
        nb = read(f, as_version=4)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for output in cell.get('outputs', []):
                if output['output_type'] == 'error':
                    return True
    return False

if __name__ == '__main__':
    file_path = sys.argv[1]
    if check_error_tags(file_path):
        sys.exit(1)
    sys.exit(0)
