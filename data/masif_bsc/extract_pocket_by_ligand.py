import numpy as np
from scipy.spatial import KDTree

def parse_pdb(pdb_file):
    atoms = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_id = line[6:11].strip()
                res_id = int(line[22:26].strip())
                x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
                atoms[(res_id, atom_id)] = {'coords': np.array([x, y, z]), 'line': line}
    return atoms

def find_nearby_residues(protein_pdb_path, ligand_pdb_path, cutoff=5.0, output=None):
    '''
    find residues within cutoff distance from ligand using KDTree
    protein_pdb_path: protein pdb file path
    ligand_pdb_path: ligand pdb file path
    cutoff: distance threshold in Å
    output: output file path
    '''
    protein_atoms = parse_pdb(protein_pdb_path)
    ligand_atoms = parse_pdb(ligand_pdb_path)

    protein_coords = np.array([atom['coords'] for atom in protein_atoms.values()])
    ligand_coords = np.array([atom['coords'] for atom in ligand_atoms.values()])

    kdtree = KDTree(protein_coords)

    nearby_residues_id = set()
    for ligand_coord in ligand_coords:
        indices = kdtree.query_ball_point(ligand_coord, cutoff) # indice is the index of protein atoms
        for idx in indices:
            id, _ = list(protein_atoms.items())[idx]
            res_id, _ = id
            nearby_residues_id.add(res_id)
    nearby_residues_id_sorted = sorted(nearby_residues_id)
    print(nearby_residues_id_sorted)

    if output:
        with open(output, 'w') as f:
            nearby_residues_line = []
            for id, atom_info in protein_atoms.items():
                if id[0] in nearby_residues_id_sorted:
                    print(f"Residue {id[0]} is within {cutoff} Å of the ligand")
                    print(atom_info['line'])
                    nearby_residues_line.append(atom_info['line'])
            f.writelines(nearby_residues_line)

