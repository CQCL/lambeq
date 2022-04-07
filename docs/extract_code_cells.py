from pathlib import Path
import nbformat as nbf


print("Extracting code from tutorial notebooks...")

files = list(x for x in Path("./tutorials").iterdir()
             if x.is_file() and x.suffix == ".ipynb")

Path("./_code").mkdir(exist_ok=True)
for file in files:
    ntbk = nbf.read(file, nbf.NO_CONVERT)
    cells_to_keep = []
    for cell in ntbk.cells:
        if cell.cell_type == "code":
            cells_to_keep.append(cell)
    new_ntbk = ntbk
    new_ntbk.cells = cells_to_keep
    nbf.write(new_ntbk, "./_code/"+file.name, version=nbf.NO_CONVERT)
