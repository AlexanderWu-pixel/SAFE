from pathlib import Path
from rdkit import Chem
from graphbpe import MolecularGraphTokenizer, MolecularSubgraphProcessor

def serialize_reparse(mol, mapping, kekule=False):
    """Track canonoica SMILES traversal back to the actual SAFE callback indices"""
    text = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=kekule)
    order = list(mol.GetPropsAsDict(True, True)['_smilesAtomOutputOrder'])
    result = Chem.MolFromSmiles(text)
    if result is None or result.GetNumAtoms() != len(order):
        raise ValueError('Atom count changed during BPE canonicalization')
    return text, result, [mapping[i] for i in order]


class BPESlicer:
    requires_kekulized_cuts = True
    def __init__(self, prefix=None):
        if prefix is None:
            project_root = Path(__file__).resolve().parent.parent
            prefix = (project_root 
                      / "data"
                      / "tokenizer"
                      / "vocab3000ring300"
                      / "pretrain-token")
        self.tokenizer = MolecularGraphTokenizer(True, 'pretrain', False)
        self.tokenizer.load(str(prefix))
        self.node_lookup = {s: i for i, s in enumerate(self.tokenizer.vocab_node)}
        self.groups = []
        self.node_ids = []
        self.requested = []

    
    def __call__(self, mol:Chem.Mol):
        # create CHem.mol object
        work = Chem.Mol(mol)
        mapping = list(range(work.GetNumAtoms()))
        for atom in work.GetAtoms():
            atom.SetAtomMapNum(0)
        if self.tokenizer.kekulize:
            Chem.Kekulize(work, clearAromaticFlags=True)
        text, work, mapping = serialize_reparse(work, mapping)
        if self.tokenizer.kekulize and work.GetNumAtoms() > 1:
            text, work, mapping = serialize_reparse(work, mapping, kekule=True)
        if self.tokenizer.kekulize:
            Chem.Kekulize(work, clearAromaticFlags=True)
        processor = MolecularSubgraphProcessor(
            text, work, kekulize=self.tokenizer.kekulize,
            simple_mode=False, allowable_rings=self.tokenizer.initial_rings,
        )
        while True:
            best, frequency = None, -1
            for candidate in processor.get_nei_smis():
                if candidate in self.node_lookup:
                    value = self.tokenizer.vocab_node_stats[candidate][1]
                    if value > frequency:
                        best, frequency = candidate, value
            if best is None:
                break
            processor.merge(best)

        self.groups, self.node_ids = [], []
        owner = {}
        for pid, atoms in processor.subgraphs.items():
            smi = processor.subgraphs_smiles[pid]
            if smi not in self.node_lookup:
                raise ValueError(f' BPE vocabulary does not cover motif: {smi}')
            
            group = sorted(mapping[i] for i in atoms)
            group_id = len(self.groups)
            for index in group:
                if index in owner:
                    raise ValueError('overlapping BPE groups')
                owner[index] = group_id
            self.groups.append(group)
            self.node_ids.append(self.node_lookup[smi])

        if set(owner) != set(range(mol.GetNumAtoms())):
            raise ValueError('BPE groups do not partition the source atoms')
        self.requested = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
            if owner[b.GetBeginAtomIdx()] != owner[b.GetEndAtomIdx()]
        ]
        return self.requested