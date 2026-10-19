
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.loader import OctaviusCatalogue, GroupCollection
    from ..data_management import SnapshotReader
    
# default libraries
from pathlib import Path

# other packages
import numpy as np
import h5py
import numba
from scipy.spatial import KDTree
from scipy.sparse import csr_array

# internal imports
from ..utils.dynamic_analyser import (
    SubsetParticleData,
    StageResult,    
)
from ..data_management import (
    SimulationData,
    ParticleStore,
    GroupStore,
    OctaviusConfig,
    OctaviusConstants,
    build_reader,
)
from ..data_management.pipeline_management import load_internals, Internals, PipelineStage

from ..log import get_logger, configure_logger

INTERNALS_FILEPATH = Path(__file__).parent.parent / "internals.yaml"



def build_cube(
    catalogue: OctaviusCatalogue,
    config: OctaviusConfig,
    snapshot_path: Path | None = None,
) -> CubeAnalyser:
    """
    Constructs a Cube

    Parameters
    ----------
    catalogue: OctaviusCatalogue
        OctaviusCatalogue object corresponding to the snapshot.
    config: OctaviusConfig
        OctaviusConfig object.
    snapshot_path: pathlib.Path or None
        Path object pointing to the original, raw snapshot (overrides config field).

    Returns
    -------
    analyser: OctaviusAnalyser
        An OctaviusAnalyser object.

    Notes
    -----
    The config can be modified from the one used to create the snapshot. Pipeline-specific parameters will be disregarded.
    """
    # initialise data structures
    constants = OctaviusConstants(mu=config.MU, frad=config.FRAD)
    snapshot_path = snapshot_path or config.snapshot_path
    reader = build_reader(snapshot_path=snapshot_path, constants=constants, config=config)
    internals = load_internals(internals_filepath=INTERNALS_FILEPATH, config=config)

    return CubeAnalyser(
        reader=reader,
        catalogue=catalogue,
        config=config,
        constants=constants,
        internals=internals,
    )


class CubeAnalyser:
    """Start of a class to handle building datacubes"""
    
    def __init__(
        self,
        reader: SnapshotReader,
        catalogue: OctaviusCatalogue,
        config: OctaviusConfig,
        constants: OctaviusConstants,
        internals: Internals,
    ) -> None:

        self._reader = reader
        self._catalogue = catalogue
        self._config = config
        self._constants = constants
        self._internals = internals

        numba.set_num_threads(config.cores_per_rank)
        configure_logger(snapshot_path=config.snapshot_path.stem, rank=0, output_level=config.terminal_output_level)

        self._collections: dict[str, GroupCollection] = {}
        if catalogue.galaxies is not None:
            self._collections["galaxies"] = catalogue.galaxies
        if catalogue.haloes is not None:
            self._collections["haloes"] = catalogue.haloes

        if not self._collections:  # high-z snapshots
            raise ValueError("Catalogue contains no group data")
        
    def compute_local_densities(
            self, group_indices: list[int] | np.ndarray, radii: list[float], group_type: str
        ) -> StageResult:
            """
            Computes the local number and mass density for the groups (of one group type) specified by group_indices.
    
            Parameters
            ----------
            group_indices: list[int]
                The indices into the Octavius catalogue of the groups to run on. You should only pass indices corresponding
                to one type of group (e.g. only galaxies or only haloes).
            group_type: str
                The type of group the indices are specified for: ``galaxies`` or ``haloes``.
    
            Returns
            -------
            result: StageResult
                A StageResult dataclass from which output columns, aligned to group_indices, can be accessed.
            """
            if group_type not in self._collections:
                raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")
            group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems
    
            stage = self._internals.stages["properties_local_environment"]
            self._verify_dependencies(stage=stage, group_type=group_type)
            data = self._collections[group_type]._data["properties"]
            columns = {}
    
            if group_type == "haloes":
                positions = data["core/minpot_pos"][:] if "core/minpot_pos" in data else data["core/com_pos_total"][:]
                masses = data["core/mass_total"][:]
            else:
                positions = data["core/com_pos_baryon"][:]
                masses = data["core/mass_baryon"][:]
    
            n_groups = len(masses)
    
            r_max = np.max(radii)
            tree = KDTree(data=positions, boxsize=self._catalogue.sim_info("boxsize", to_units="kpc"))
            sdm = tree.sparse_distance_matrix(
                other=tree, max_distance=r_max, output_type="coo_array"
            )  # if your scipy is pre-1.18, use "coo_matrix"
            vol_prefactor = 4.0 / 3.0 * np.pi
    
            for radius in radii:
                in_range = (
                    sdm.data <= radius
                )  # NOTE: the matrix does include self-distance 0 as we passed other=tree ^, I verified this manually with test data
                valid_array = np.ones(in_range.sum())
                adj = csr_array(
                    arg1=(valid_array, (sdm.row[in_range], sdm.col[in_range])), shape=(n_groups, n_groups)
                )  # unhelpful argument name
    
                volume = vol_prefactor * radius**3
                local_mass_density = (adj @ masses / volume)[
                    group_indices
                ]  # adjacency matrix algebra gets the quantities in a vectorised way
                local_number_density = (adj @ np.ones(shape=n_groups) / volume)[group_indices]
    
                columns[f"local_mass_density_{radius:.0f}kpc"] = local_mass_density
                columns[f"local_number_density_{radius:.0f}kpc"] = local_number_density
    
            result = StageResult(
                group_type=group_type,
                group_indices=group_indices,
                columns=columns,
            )
    
            return result
        
    def probe_self_data(
        self, group_type: str, group_indices: list[int] | np.ndarray
    ):
        if group_type not in self._collections:
            raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems

        stage = self._internals.stages["properties_local_environment"]
        self._verify_dependencies(stage=stage, group_type=group_type)
        data = self._collections[group_type]._data["properties"]
        columns = {}
        
        print(f"Keys in data: {data.keys()}")
        print(f"Keys in data['core']: {data['core'].keys()}")

        if group_type == "galaxies":
            positions = data["core/com_pos_baryon"][:]
            masses = data["core/mass_baryon"][:]
        else:
            raise ValueError(f"ill do '{group_type}' at some point")            
        
        n_groups = len(masses)

        r_max = np.max(radii)
        tree = KDTree(data=positions, boxsize=self._catalogue.sim_info("boxsize", to_units="kpc"))
        sdm = tree.sparse_distance_matrix(
            other=tree, max_distance=r_max, output_type="coo_array"
        )  # if your scipy is pre-1.18, use "coo_matrix"
        vol_prefactor = 4.0 / 3.0 * np.pi

        for radius in radii:
            in_range = (
                sdm.data <= radius
            )  # NOTE: the matrix does include self-distance 0 as we passed other=tree ^, I verified this manually with test data
            valid_array = np.ones(in_range.sum())
            adj = csr_array(
                arg1=(valid_array, (sdm.row[in_range], sdm.col[in_range])), shape=(n_groups, n_groups)
            )  # unhelpful argument name

            volume = vol_prefactor * radius**3
            local_mass_density = (adj @ masses / volume)[
                group_indices
            ]  # adjacency matrix algebra gets the quantities in a vectorised way
            local_number_density = (adj @ np.ones(shape=n_groups) / volume)[group_indices]

            columns[f"local_mass_density_{radius:.0f}kpc"] = local_mass_density
            columns[f"local_number_density_{radius:.0f}kpc"] = local_number_density

        result = StageResult(
            group_type=group_type,
            group_indices=group_indices,
            columns=columns,
        )

        return result
    

        
        
        
        
        
        
        
    
        

    
    def _extract_groups(
            self,
            group_type: str,
            group_indices: np.ndarray,
            ptype_columns: dict[str, list[str]],
        ) -> dict[str, SubsetParticleData]:
            """
            Extracts groups from the catalogue and then their particle-level requisite data. Returns:

            - result: a dictionary of SubsetParticleData classes, keyed by ptype.
            """
            membership_group = self._collections[group_type]._data["membership"]  # NOTE: hardcoded, update if changing
            assert isinstance(membership_group, h5py.Group)

            result: dict[str, SubsetParticleData] = {}

            for ptype, columns in ptype_columns.items():
                result[ptype] = self._extract_ptype(
                    ptype=ptype, membership_group=membership_group, group_indices=group_indices, columns=columns
                )

            return result

    def _extract_ptype(
            self,
            ptype: str,
            membership_group: h5py.Group,
            group_indices: np.ndarray,
            columns: list[str],
        ) -> SubsetParticleData:
            """
            Extracts data for requested columns for the specified ptype. Returns:
    
            - result: a SubsetParticleData for the requested ptype.
            """
            indices_dataset = membership_group[f"{ptype}_indices"]  # slab-read indices (large)
            offsets = membership_group[f"{ptype}_offsets"]  # full read offsets
    
            starts = offsets[group_indices]
            ends = offsets[group_indices + 1]
            lengths = ends - starts  # to make code cleaner
            total_particles = np.sum(lengths)
    
            # allocate outputs
            particle_indices = np.empty(total_particles, dtype=np.int64)
            new_offsets = np.empty(len(group_indices) + 1, dtype=np.int64)  # not snapshot aligned
    
            # construct new offsets from old offsets masked by group indices
            new_offsets[0] = 0
            np.cumsum(lengths, out=new_offsets[1:])
    
            # loop over groups, collect their particles
            position = 0  # positional index
            for i in range(len(group_indices)):
                if lengths[i] > 0:  # need this because a group can miss a ptype
                    particle_indices[position : position + lengths[i]] = indices_dataset[starts[i] : ends[i]]
    
                position += lengths[i]
    
            # sort indices first (must be increasing)
            sort_order = np.argsort(particle_indices, stable=True)
            sorted_indices = particle_indices[sort_order]
    
            column_data = self._reader.read_requested_columns(
                ptype=ptype,
                datasets=columns,
                sorted_snapshot_indices=sorted_indices,
            )
    
            # then unsort back to group order (necessary for sorted_idx in pipeline stages)
            unsort_order = np.argsort(sort_order)
            for col in column_data:
                column_data[col] = column_data[col][unsort_order]
    
            data = SubsetParticleData(columns=column_data, offsets=new_offsets)
    
            return data

    def _verify_dependencies(
            self,
            stage: PipelineStage,
            group_type: str,
        ) -> None:
            """
            Verifies the analysis can be done from what exists in the catalogue (stage dependencies).
            """
            data = self._collections[group_type]._data
    
            for dependent in stage.requires:
                dependent_label = self._internals.stages[dependent].label
    
                if f"properties/{dependent_label}" not in data:
                    raise KeyError(f"Dependent stage {dependent_label} is not present in the catalogue.")
                
    
    