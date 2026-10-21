import numpy as np
import torch
from torch import Tensor 

from rdkit import Chem 

from eftalfm.data.utils import get_distances

def map_canonical_atom_order(mol, addHs:bool=False):
    Chem.SanitizeMol(mol)

    if addHs:
        # Convert mol to its canonical SMILES and then back to a molecule
        # canonical_mol = Chem.MolFromSmiles(Chem.CanonSmiles(Chem.addHs(Chem.MolToSmiles(mol))))
        canonical_mol = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True)))
    else:
        canonical_mol = Chem.MolFromSmiles(Chem.CanonSmiles(Chem.MolToSmiles(mol)))
     
    
    Chem.SanitizeMol(canonical_mol)
    # Get the mapping -- ith canonical atom idx: original atom idx
    atom_map = list(mol.GetSubstructMatch(canonical_mol))
    if len(atom_map) == 0:
        atomlist = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        canonical_atomlist = [atom.GetAtomicNum() for atom in canonical_mol.GetAtoms()]

        n_mol_atom_num = len(atomlist)
        if np.all(np.array(canonical_atomlist[n_mol_atom_num:])==1):
            inv_mapping = list(canonical_mol.GetSubstructMatch(mol))
            # mapping is a list of indices such that a1[i] = a2[mapping[i]]
            full_atom_map = [-1] * len(canonical_atomlist)  # Initialize with a placeholder value.
            for i, j in enumerate(inv_mapping):
                full_atom_map[j] = i

            atom_map = [x for x in full_atom_map if x != -1]
            added_new = True 
        else:
            raise ValueError(f"canonical={canonical_atomlist} does not match the non H in atomlist={atomlist}")
    else:    
        added_new = False 
    return atom_map, added_new

 
def get_rotation_matrix(X:Tensor, Y:Tensor, tol=1e-7):
    """Get the rotation matrix XR = Y
    Use the first three nodes to determine R
    Make sure the first three nodes provide a nondegenerate basis! 

    The two bases are constructed as [A1, A2, A1xA2] and [B1, B2, B1xB2], 
    so both are right-handed whenever A1, A2 (and B1, B2) are linearly independent. 
    Working through the algebra: det(R) = det(X_prime) / det(Y_prime) = |A1xA2|² / |B1xB2|², 
    which is strictly positive for any non-degenerate input. 
    So in exact arithmetic det(R) < 0 is impossible here — it can only mean (a) the upstream basis 
    filter let through a near-collinear triple, (b) the caller fed in points that aren't actually 
    related by a rigid motion, or (c) numerical breakdown. 
    None of those are "a reflection that should be silently fixed."

    On naive-vs-Kabsch, for an exact 3-anchor problem the direct solve is exact and 
    gives you a single deterministic answer; Kabsch's SVD-based projection is the right tool 
    when you have noise or >3 points and want a least-squares fit, neither of which applies here. 
    The only real argument for switching would be code simplicity (one path instead of two).


    X: (n, 3)
    Y: (n, 3)
    
    Test cases:
        X = torch.tensor([
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.0, 3.0, 3.0]
        ])

        Y = torch.tensor([
            [4.0, 5.0, 6.0],
            [4.0, 6.0, 6.0],
            [5.0, 5.0, 6.0]
        ])

        R = get_rotation_matrix(X, Y)
        print("Rotation matrix R:")
        print(R)

        # Verify rotation
        Y_computed = (X - X[0]) @ R + Y[0]

        # Use the same rotation matrix to check the following 
        positions = torch.normal(0,1,(10,3), dtype=torch.float32)
        positions_moved = positions - positions[0]
        R_check = get_rotation_matrix(positions_moved, positions_moved @ R)

    """   
    # Given original points
    A1 = X[1] - X[0]
    A2 = X[2] - X[0]

    # Given rotated points
    B1 = Y[1] - Y[0]
    B2 = Y[2] - Y[0]

    # Compute the cross-products to form complete bases
    A3 = torch.cross(A1, A2, dim=-1)
    B3 = torch.cross(B1, B2, dim=-1)

    # Construct the full matrices X' and Y'
    X_prime = torch.column_stack((A1, A2, A3))
    Y_prime = torch.column_stack((B1, B2, B3))

    # Compute the rotation matrix R
    R = torch.linalg.solve(Y_prime, X_prime, left=False)
    
    # The constructed bases [A1, A2, A1 x A2] and [B1, B2, B1 x B2]
    # are right-handed when non-degenerate, so the solved map should
    # already be a proper rotation. A negative determinant indicates
    # degeneracy or misuse of this helper, not a reflection to repair.
    det_R = torch.det(R)
    if det_R < 0:
        raise ValueError(f"naive rotation solve produced det(R)={det_R:.6f} — basis is degenerate or inputs are not related by a rigid motion")

    
    # Translation vector: 1 by d 
    t = Y.mean(dim=0, keepdim=True) - X.mean(dim=0, keepdim=True) @ R

    val_check = torch.max(torch.abs(X@R+t-Y))
    val_check_R = torch.max(torch.abs(R @ R.T - torch.eye(3)))
    if val_check > tol and val_check_R > tol:
        raise ValueError(f"max(abs(XR - Y)) = {val_check:.6f}, max(abs(R @ R.T - I)) = {val_check_R:.6f}")
    
    return R, t

            
def align_nondegenerate_basis(
    positions: Tensor,
    force_no_change: bool = False,
    tol: float = 1e-7,
    eps: float = 1e-8,
):
    """Find a fixed-order non-degenerate basis without permuting atom rows.
    Corner cases:
        positions = torch.from_numpy(np.random.normal(0,1,(5,3)))
        positions[1] = positions[0]
        positions[2] = positions[0]
    """
    n_nodes = positions.shape[0]
    # Kept for backward-compatible call sites. The amended alignment never
    # permutes atom rows; it may choose a later anchor atom if earlier anchors
    # are coincident or too close to collinear.
    _ = force_no_change
    
    ## Pick three points that are not degenerated.
    # The first one must be at (0,0,0).
    A0 = positions[0] 
    displacements = positions - A0

    axis_idx = None
    for j in range(1, n_nodes):
        if bool(torch.linalg.vector_norm(displacements[j], dim=-1) > eps):
            axis_idx = j
            break

    if axis_idx is None:
        return False, None

    vec_A1 = displacements[axis_idx]
    norm_A1 = torch.linalg.vector_norm(vec_A1, dim=-1)
    # Then find another point that does not align with the axis.
    for k in range(1, n_nodes):
        if k == axis_idx:
            continue
        vec_A2 = displacements[k]
        norm_A2 = torch.linalg.vector_norm(vec_A2, dim=-1)

        # Compute cross product
        cross_product = torch.cross(vec_A1, vec_A2, dim=-1)
        area = torch.linalg.vector_norm(cross_product, dim=-1)
        rel_scale = torch.clamp(norm_A1 * norm_A2, min=1.0)

        # Check if the cross product is meaningfully non-zero.
        if bool(area > max(tol, eps) * rel_scale):
            return True, (0, axis_idx, k)
    return False, None


def _align_collinear_positions_to_x_axis(
    positions_moved: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Place a fixed-order collinear geometry on x using signed projection."""
    positions_final = torch.zeros_like(positions_moved)
    if positions_moved.numel() == 0:
        return positions_final

    distances_from_origin = torch.linalg.vector_norm(positions_moved, dim=-1)
    nonzero = torch.nonzero(distances_from_origin > eps, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return positions_final

    axis_idx = int(nonzero[0].item())
    axis = positions_moved[axis_idx] / distances_from_origin[axis_idx]
    positions_final[:, 0] = positions_moved @ axis
    return positions_final


def align_positions(positions:Tensor, centered:bool=False, naive:bool=True, tol:float=1e-5, eps=1e-8, force_no_basis_change:bool=False):
    """Align the positions in a systematic way:
    Step 0: Get a fixed-order nondegenerated basis (0,j,k)
    Step 1: Move the first node to the origin 
    Step 2: Move axis node j to (|pj-p0|, 0, 0), keeping the distance
            between the first and the axis node stays the same
    Step 3: Move the k-th node to (x_k0, 0, x_k2) where x_k2>0 by solving
                          x_k0^2 + x_k2^2 = |pk-p0|^2
                (x_k0-|pj-p0|)^2 + x_k2^2 = |pk-pj|^2
    Step 4: Compute the rotation matrix based on the (0,j,k) nodes
    Step 5: Rotate the whole set of nodes to the final positions 
    
    OUTPUT:
    =====================================================================
    positions: (n,3)
    """
    assert positions.dim() == 2 and positions.size(-1) == 3, \
        "positions must be a 2D tensor of shape [n_atom, 3]"

    n_atom = positions.size(0)
    if n_atom < 2:
        # If there is only 1 atom or none, there's nothing to align meaningfully.
        # But let's at least return a shifted version (the first node at origin).
        print(f"Less than 2 atoms in the molecule: {n_atom}. No meaningful alignment.")
        return positions - positions[0] if n_atom > 0 else positions
     
    # ------------------------------------------------------
    #  STEP 0: Find a non-degenerate basis without permuting rows.
    # ------------------------------------------------------
    # If earlier anchors are coincident or nearly collinear, use the first
    # later atom that defines a stable axis/plane.
    if n_atom >= 3:
        ## Make sure the first three nodes form a nondegenerated basis
        to_align, basis_indices = align_nondegenerate_basis(
            positions,
            force_no_change=force_no_basis_change,
            tol=tol,
            eps=eps,
        )
 
    # ------------------------------------------------------
    #  STEP 1: Move the first node to the origin.
    # ------------------------------------------------------
    positions_moved = positions - positions[0]

    # We'll build an intermediate set of "target" coordinates
    # for the selected non-degenerate triple.
    positions_inter = torch.zeros_like(positions_moved)

    # ------------------------------------------------------
    #  If we only have 2 atoms, we are done -- everything is on x-axis.
    # ------------------------------------------------------
    if n_atom == 2:
        positions_final = _align_collinear_positions_to_x_axis(positions_moved, eps=eps)
    else:
        # n_atom >= 3
        if not to_align:
            # Degenerate or collinear case: use signed projection onto the
            # first non-zero canonical axis so atoms on opposite sides do not collapse.
            positions_final = _align_collinear_positions_to_x_axis(positions_moved, eps=eps)
        else:
            # ------------------------------------------------------
            #  STEP 2: Place axis atom j at (|pj-p0|, 0, 0).
            #  STEP 3: Place atom k at (x_k0, 0, x_k2) with x_k2 > 0.
            # ------------------------------------------------------
            _, ij, ik = basis_indices
            # Distances
            d_0j = torch.linalg.vector_norm(positions_moved[ij] - positions_moved[0], dim=-1)
            d_0k = torch.linalg.vector_norm(positions_moved[ik] - positions_moved[0], dim=-1)
            d_jk = torch.linalg.vector_norm(positions_moved[ik] - positions_moved[ij], dim=-1)
            positions_inter[ij, 0] = d_0j
            ## Move the third node to (x_k0, 0, x_k2) where x_k2>0 
            positions_inter[ik, 0] = (d_0j**2 - d_jk**2 + d_0k**2)/2/(d_0j + eps)
            positions_inter[ik, 2] = torch.sqrt(torch.clamp(d_0k**2 - positions_inter[ik, 0]**2, min=0.0))
            
            # ------------------------------------------------------
            #  STEP 4: Compute the rotation matrix that sends
            #          positions_moved[[0,j,k]] --> positions_inter[[0,j,k]]
            # ------------------------------------------------------
            # 'get_rotation_matrix(A, B)' is assumed to return a rotation R
            # s.t. A @ R ~= B.
            basis = torch.tensor([0, ij, ik], device=positions_moved.device, dtype=torch.long)
            R, _ = get_rotation_matrix(positions_moved[basis],
                                       positions_inter[basis],
                                       naive=naive, tol=tol)
            # ------------------------------------------------------
            #  STEP 5: Rotate the whole set
            # ------------------------------------------------------
            positions_final = positions_moved @ R
    
    # ------------------------------------------------------
    #  Optional: if `centered=True`, you might want to re-center
    #  the final structure to e.g. the centroid or keep it with p0 at origin.
    # ------------------------------------------------------
    pairwise_dist_final = get_distances(positions_final)['pairwise_distances'].squeeze(-1)
    pairwise_dist       = get_distances(positions)['pairwise_distances'].squeeze(-1)
    curr_tol =  torch.max(torch.abs(pairwise_dist_final - pairwise_dist))
    if curr_tol > tol:
        raise ValueError(
            f"Distance mismatch = {curr_tol:.6f} exceeds tolerance = {tol:.6f}!"
        )
    if centered: 
        positions_final = positions_final - torch.mean(positions_final, dim=0, keepdim=True)
    return positions_final 
    

if __name__ == "__main__":
    positions = torch.normal(0,1,(10,3)).to(torch.float32)
    positions_final, R = align_positions(positions, tol=1e-5)
