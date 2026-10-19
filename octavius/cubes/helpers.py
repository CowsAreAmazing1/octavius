
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
from ..cubes.density_cube import DensityCubeData

from ..utils.dynamic_analyser import (
    OctaviusAnalyser,
    _build_particle_stores,
    _build_group_store,
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


class CubeAnalyser(OctaviusAnalyser):
    """Start of a class to handle building datacubes"""
  
    # def probe_self_data(
    #     self, group_type: str, group_indices: list[int] | np.ndarray
    # ):
    #     if group_type not in self._collections:
    #         raise KeyError(f"Group type '{group_type}' is not present in the catalogue.")
    #     group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))  # sort to avoid h5py problems

    #     stage = self._internals.stages["properties_local_environment"]
    #     self._verify_dependencies(stage=stage, group_type=group_type)
    #     data = self._collections[group_type]._data["properties"]
    #     columns = {}
        
    #     print(f"Keys in data: {data.keys()}")
    #     print(f"Keys in data['core']: {data['core'].keys()}")

    #     if group_type == "galaxies":
    #         positions = data["core/com_pos_baryon"][:]
    #         masses = data["core/mass_baryon"][:]
    #     else:
    #         raise ValueError(f"ill do '{group_type}' at some point")            
        
    #     n_groups = len(masses)

    #     r_max = np.max(radii)
    #     tree = KDTree(data=positions, boxsize=self._catalogue.sim_info("boxsize", to_units="kpc"))
    #     sdm = tree.sparse_distance_matrix(
    #         other=tree, max_distance=r_max, output_type="coo_array"
    #     )  # if your scipy is pre-1.18, use "coo_matrix"
    #     vol_prefactor = 4.0 / 3.0 * np.pi

    #     for radius in radii:
    #         in_range = (
    #             sdm.data <= radius
    #         )  # NOTE: the matrix does include self-distance 0 as we passed other=tree ^, I verified this manually with test data
    #         valid_array = np.ones(in_range.sum())
    #         adj = csr_array(
    #             arg1=(valid_array, (sdm.row[in_range], sdm.col[in_range])), shape=(n_groups, n_groups)
    #         )  # unhelpful argument name

    #         volume = vol_prefactor * radius**3
    #         local_mass_density = (adj @ masses / volume)[
    #             group_indices
    #         ]  # adjacency matrix algebra gets the quantities in a vectorised way
    #         local_number_density = (adj @ np.ones(shape=n_groups) / volume)[group_indices]

    #         columns[f"local_mass_density_{radius:.0f}kpc"] = local_mass_density
    #         columns[f"local_number_density_{radius:.0f}kpc"] = local_number_density

    #     result = StageResult(
    #         group_type=group_type,
    #         group_indices=group_indices,
    #         columns=columns,
    #     )

    #     return result
    
    def get_useful_density_cube_data(
        self,
        group_indices: list[int] | np.ndarray,
    ):
        assert self._reader.simulation_attributes is not None
        
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
        group_type = "galaxies"
        if group_type not in self._collections:
                    raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        
        ptype_columns = {
            'gas': [
                'dust_mass', 'mass', 'metallicity', 'pos', 'smoothing_length'
            ], 
            'star': [
                'mass', 'pos', 'vel' # 'metallicity', 'age', 
            ]
        }
        
        gal_reqs = {"star": ptype_columns["star"], "gas": ptype_columns["gas"]}
        gal_data = self._extract_groups(group_type=group_type, group_indices=group_indices, ptype_columns=gal_reqs)
        
        particles = _build_particle_stores(subset_data=gal_data, internals=self._internals)
        galaxies = _build_group_store(
            group_type="galaxies", group_indices=group_indices, subset_data=gal_data, internals=self._internals
        )
        
        groups = {"galaxies": galaxies}
        
        sim_data = SimulationData(
            simulation=self._reader.simulation_attributes,
            constants=self._constants,
            particles=particles,
            groups=groups,
        )
        
        return sim_data

    def wrangle_density_data(
        self, simulation_data: SimulationData, config: OctaviusConfig
    ):
        stars = simulation_data.particles["star"]
        
        print(f"groups keys : {simulation_data.groups.keys()}")
        print(f"galaxies columns : {simulation_data.groups["galaxies"].columns}")
        
        return DensityCubeData(
            pos=stars["pos"],
            mass=stars["mass"],
            # centre=simulation_data.groups["galaxies"]["core/com_pos_baryon"]
        )
