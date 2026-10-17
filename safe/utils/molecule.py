from rdkit import Chem

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def convert_to_canonical(list_of_smiles):
    canonical_smiles_list = []
    for smiles in list_of_smiles:
        mol = get_mol(smiles)
        if mol is None:
            canonical_smiles_list.append(smiles)
        else:
            canonical_smiles_list.append(Chem.MolToSmiles(mol))
    return canonical_smiles_list

def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def convert_graph_to_smiles(generated, tokenizer, excluded_smiles):
    valid_list = []
    num_components = []
    all_smiles = []
    valid_unique = []
    for graph_idx, graph in enumerate(generated):
        node_types, bond_adj, position_adj = graph
        bond_adj = bond_adj - 1 # ensure null edge == -1
        position_adj = position_adj - 1 # ensure null edge == -1

        smiles1, smiles2, _ = tokenizer.decode(
            node_types.tolist(), 
            bond_adj.tolist(), 
            position_adj.tolist()
        )
        # mol = get_mol(smiles1) or get_mol(smiles2)
        mol = get_mol(smiles1)
        if mol is None:
            all_smiles.append(None)
            valid_unique.append(None)
            continue
        smiles = mol2smiles(mol)
        if smiles:
            components = smiles.split('.')
            num_components.append(len(components))
            largest_smiles = max(components, key=len)
            
            largest_mol = Chem.MolFromSmiles(largest_smiles)
            largest_smiles = mol2smiles(largest_mol)
            valid_list.append(largest_smiles)
            all_smiles.append(largest_smiles)
            if largest_smiles not in excluded_smiles:
                valid_unique.append(largest_smiles)
            else:
                valid_unique.append(None)
        else:
            all_smiles.append(None)
            valid_unique.append(None)

    return valid_list, valid_unique, all_smiles


def convert_graph_to_mol(generated, tokenizer):
    all_mols = []
    all_smiles = []
    for graph_idx, graph in enumerate(generated):
        node_types, bond_adj, position_adj = graph
        bond_adj = bond_adj - 1 # ensure null edge == -1
        position_adj = position_adj - 1 # ensure null edge == -1

        mol, smiles = tokenizer.decode(
            node_types.tolist(), 
            bond_adj.tolist(), 
            position_adj.tolist(),
            return_mol=True
        )
        all_mols.append(mol)
        all_smiles.append(smiles)
    return all_mols, all_smiles

def check_valid(smiles):
    """
    Check if a SMILES string is valid.
    """
    return Chem.MolFromSmiles(smiles) is not None