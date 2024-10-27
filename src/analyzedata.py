from rdkit import Chem

# Load molecules from the SDF file
supplier = Chem.SDMolSupplier('../data/structures.sdf')

molecules = [mol for mol in supplier if mol is not None]


print(f"{len(molecules)}")
