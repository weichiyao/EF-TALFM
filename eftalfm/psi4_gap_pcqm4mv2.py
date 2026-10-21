from __future__ import annotations
import os
from contextlib import contextmanager

import numpy as np 
import psi4
from rdkit import Chem
from rdkit.Chem import Descriptors 

HARTREE_TO_EV = 27.211386245988

def mol2xyz(mol: Chem.Mol) -> str:
    conformer = mol.GetConformer()
    lines = []
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol()} {pos.x:.16f} {pos.y:.16f} {pos.z:.16f}"
        )
    lines.append("units angstrom")
    return "\n".join(lines)


def get_spin_multiplicity(mol: Chem.Mol) -> int:
    mol.UpdatePropertyCache(strict=False)
    num_radical_electrons = Descriptors.NumRadicalElectrons(mol)
    return int(num_radical_electrons) + 1


def _frontier_orbital_energies(wfn, orbital_channel: str) -> tuple[float, float]:
    """Select the highest occupied and lowest virtual energies in Hartree."""
    epsilon_subset = (wfn.epsilon_a_subset if orbital_channel == "alpha"
                      else wfn.epsilon_b_subset)
    occupied = epsilon_subset("AO", "OCC").np
    virtual = epsilon_subset("AO", "VIR").np
    if len(occupied) == 0 or len(virtual) == 0:
        raise ValueError(
            f"Cannot extract {orbital_channel} gap with {len(occupied)} "
            f"occupied and {len(virtual)} virtual orbitals."
        )
    if not np.isfinite(occupied).all() or not np.isfinite(virtual).all():
        raise ValueError("Psi4 returned nonfinite orbital energies.")
    # Select by occupation, rather than assuming the globally sorted ALL
    # array remains Aufbau-ordered across symmetry blocks/spin channels.
    return float(np.max(occupied)), float(np.min(virtual))



@contextmanager
def _fresh_psi4(
    memory: str = "1 GB",
    n_threads: int = 16,
    scratch: str = "/tmp/psi4_scratch",
    quiet: bool = True,
):
    os.makedirs(scratch, exist_ok=True)
    psi4.core.clean_options()

    if quiet:
        psi4.core.set_output_file(os.devnull, False)

    psi4.core.IOManager.shared_object().set_default_path(scratch)
    psi4.set_num_threads(n_threads)
    psi4.set_memory(memory)

    try:
        yield
    finally:
        psi4.core.IOManager.shared_object().psiclean()
        psi4.core.clean_timers()
        psi4.core.clean()

def get_psi4_gap(
    mol: Chem.Mol,
    charge: int | None= None,
    spin_multiplicity: int | None = None,
    method: str = "B3LYP/6-31G(d)",
    orbital_channel: str = "alpha",
    memory: str = "1 GB",
    num_threads: int = 16,
    scratch_folder: str = "/tmp/psi4_scratch",
    scf_type: str = "df",
    df_basis_scf: str | None = None,
    use_disk_df: bool = False,
    quiet: bool = True,
    do_optimize: bool = False,
    reference: str | None = None,
) -> float | dict:
    
    if mol is None:
        raise ValueError("RDKit molecule is None.")
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit molecule does not contain a conformer.")
    
    if charge is None:
        charge = int(Chem.GetFormalCharge(mol))
    if spin_multiplicity is None:
        spin_multiplicity = get_spin_multiplicity(mol)
    
    charge, spin_multiplicity = int(charge), int(spin_multiplicity)

    xyz_body = mol2xyz(mol)
    xyz_block = f"{charge} {spin_multiplicity}\n{xyz_body}"
    
    if reference is None:
        reference = "rks" if spin_multiplicity == 1 else "uks"
    try:
        functional, basis = method.split("/", 1)
    except ValueError as exc:
        raise ValueError(
            "method must be formatted like 'B3LYP/6-31G(d)'."
        ) from exc

    with _fresh_psi4(memory, num_threads, scratch_folder, quiet=quiet):
        psi_mol = psi4.geometry(xyz_block)
        options = {
            "basis": basis,
            "reference": reference,
            "scf_type": scf_type,
            "DF_INTS_IO": "SAVE" if use_disk_df else "NONE",
            "guess": "sad",
        }
        if df_basis_scf is not None:
            options["df_basis_scf"] = df_basis_scf
        
        psi4.set_options(options)
        
        if do_optimize:
            _, scf_wfn = psi4.optimize(functional, molecule=psi_mol, return_wfn=True)
        else:
            _, scf_wfn = psi4.energy(functional, molecule=psi_mol, return_wfn=True)

        homo_hartree, lumo_hartree = _frontier_orbital_energies(scf_wfn, orbital_channel)
        gap_hartree = lumo_hartree - homo_hartree
        gap_ev = gap_hartree * HARTREE_TO_EV
        return gap_ev
       
