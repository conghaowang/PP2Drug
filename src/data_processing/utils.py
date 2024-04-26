from rdkit import Chem
import numpy as np
import sys
# sys.path.append('../')
from openbabel import pybel
import random
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
# from IPython.display import SVG, Image
# IPythonConsole.drawOptions.addAtomIndices = True
# IPythonConsole.ipython_useSVG = True
# IPythonConsole.molSize = 600, 300

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_TYPE_MAPPING = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7}
PP_TYPE_MAPPING = {
    'Linker': 0,
    'Hydrophobic': 1,
    'Aromatic': 2,
    'Cation': 3,
    'Anion': 4,
    'HBond_donor': 5,
    'HBond_acceptor': 6,
    'Halogen': 7
}
MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = int(random.random() * n)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])

    return Psample


def group_by(mol, ligand, level='pp'):
    # this function is used to highlight the atoms from different pharmachophores / clusters with different colors in the ligand
    my_cmap = matplotlib.colormaps['coolwarm']
#     my_cmap = cm.get_cmap('coolwarm')
    if level == 'cluster':
        n_group = len(ligand.graph.node_clusters)
    elif level == 'pp':
        n = 0
        for cluster in ligand.graph.node_clusters:
            n += cluster.positions.shape[1]
        n_group = n
        pp_id = 0
        
    my_norm = Normalize(vmin=0, vmax=n_group)
    atommap, bondmap = {}, {}
    for i in range(len(ligand.graph.node_clusters)):
        cluster = ligand.graph.node_clusters[i]
        for node in cluster.nodes:
#             node = cluster.nodes[pp_id]
            atom_idx = node.atom_indices
            if level == 'cluster':
                for atom_id in atom_idx:
                    atom = mol.GetAtoms()[atom_id]
                    atom.SetProp("atomNote", str(i))
                atommap.update({atom_id:my_cmap(my_norm(i))[:3] for atom_id in atom_idx})
            elif level == 'pp':
                for atom_id in atom_idx:
                    atom = mol.GetAtoms()[atom_id]
                    atom.SetProp("atomNote", str(pp_id))
                atommap.update({atom_id:my_cmap(my_norm(pp_id))[:3] for atom_id in atom_idx})
                pp_id += 1
                
    highlights = {
        "highlightAtoms": list(atommap.keys()),
        "highlightAtomColors": atommap,
        "highlightBonds": list(bondmap.keys()),
        "highlightBondColors": bondmap,
    }
    # mol_ = rdMolDraw2D.PrepareMolForDrawing(mol)
    # imgsize = (600, 300)
    # drawer = rdMolDraw2D.MolDraw2DSVG(*imgsize)
    # drawer.DrawMolecule(mol_, **highlights)
    # drawer.FinishDrawing()
    # svg = drawer.GetDrawingText()
    # display(SVG(svg.replace('svg:','')))
    return highlights