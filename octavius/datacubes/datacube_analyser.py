"""
Internal functions for building datacubes, using existing methods from the `OctaviusAnalyser` class.
Handles access to all data avaliable in the catalogue, snapshot and config.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.loader import OctaviusCatalogue
    from ..data_management import OctaviusConfig

# default packages
from pathlib import Path

# other packages
import numpy as np
from h5py._hl.group import Group
from h5py._hl.dataset import Dataset

# internal imports
from .density_cube import DensityCubeData
from ..utils.dynamic_analyser import (
    OctaviusAnalyser,
    _build_particle_stores,
)
from ..data_management import (
    OctaviusConstants,
    build_reader,
)
from ..data_management.pipeline_management import load_internals, Internals, PipelineStage

INTERNALS_FILEPATH = Path(__file__).parent.parent / "internals.yaml"

class CubeAnalyser(OctaviusAnalyser):
    """Start of a class to handle building datacubes"""
    
    def splat_keys(
        self,
        group_type: str,
    ):
        assert self._reader.simulation_attributes is not None
        
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        # All property data is in 'properties'; other keys store membership data, etc.
        group_data = self._collections[group_type]._data['properties']
        assert isinstance(group_data, Group)

        keys_1 = list(group_data.keys())
        print(f"Avaliable keys 1: {keys_1}")
        
        for key_1 in keys_1:
            data_1 = group_data[key_1]
            assert isinstance(data_1, Group) # Directory
            
            keys_2 = list(data_1.keys())
            print(f"Avaliable keys 2: {keys_2}")
            
            for key_2 in keys_2:
                data_2 = data_1[key_2]
                assert isinstance(data_2, Dataset) # File
                
                print(f"Size of {key_1}/{key_2}: {data_2.shape}")
            
            
            
    def get_manual_property_data(
        self,
        group_type: str,
        property_name: str,
    ) -> np.ndarray:
        assert self._reader.simulation_attributes is not None
        
        
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        # All property data is in 'properties'; other keys store membership data, etc.
        group_data = self._collections[group_type]._data['properties']
        assert isinstance(group_data, Group)
        
        property_data = group_data[property_name]
        assert isinstance(property_data, Dataset)
        
        return property_data[:] # "coerse" to numpy array
        
        
        
        
    
    def get_spatial_cube_data(
        self,
        group_type: str,
        group_indices: list[int] | np.ndarray,
        property_name: str, 
        ptype: str =  "gas",
        alignment: str | None = None,
    ):
        """
        Collects required data from the catalogue and snapshot to build a spatial datacube for the desired property, for the for the specified galaxies/haloes.
        
        Parameters
        ----------
        group_type: str
            The type of group to run on. Must be present in the catalogue.
        group_indices: list[int]
            The indices into the Octavius catalogue of the galaxies to run on.
        property_name: str
            The name of the property to build the datacube for.
            like `particle_specific/sfr` or `core/kappa_rot_gas`
        ptype: str
            The particle type to use. Must be present in the snapshot. Default: `gas`
        alignment: str | None
            The alignment method to use. TODO allow alignment with angular momentum. Im imagining that this will flatten the object into the xy plane.
            Also just allow a general alignment/rotation to be passed, if thats useful.

        Returns
        -------
        result: DensityCubeData
            A `DensityCubeData` dataclass from 
        """
        assert self._reader.simulation_attributes is not None
        
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        group_data = self._collections[group_type]._data['properties']
        assert isinstance(group_data, Group)
        
        # Center of the cube in physical coordinates
        com_pos_baryon = group_data["core/com_pos_baryon"]
        assert isinstance(com_pos_baryon, Dataset)
        coms = com_pos_baryon[group_indices]
        mean_com = np.mean(coms, axis=0)
        
        if alignment is not None:
            raise NotImplementedError("hold on")
        
        if ptype != "gas":
            # No idea how to handle things that arent gas particles yet. From what i can
            # see, gas particles are the only ones that have a smoothing length property,
            # so i guess the only ones that are SPHly modelled?
            raise NotImplementedError("Only gas particles are currently supported.")
        
        spatial_columns = {ptype : [
            "mass",
            "pos",
            "smoothing_length", # NOTE: Only gas has "smoothing_length" here
        ]}
        spatial_subset_data = self._extract_groups(
            group_type=group_type,
            group_indices=group_indices, 
            ptype_columns=spatial_columns,
        )    
        particles = _build_particle_stores(
            subset_data=spatial_subset_data,
            internals=self._internals
        )
        
        spatial_data = particles[ptype]
        
        property_data = self.get_manual_property_data(
            group_type=group_type,
            property_name=property_name,
        )
        
        return DensityCubeData(
            pos=spatial_data['pos'],
            mass=spatial_data['mass'],
            property=property_data,
            smoothing_length=spatial_data['smoothing_length'],
            com=mean_com,
            angular_momentum=None,
            config=self._config,   
        )
    
    # TODO: These functions are obviously not very general currently
    def get_useful_density_cube_data(
        self,
        group_type: str,
        group_indices: list[int] | np.ndarray,
        alignment: str | None = None,
        ptype: str = "gas",
        smoothing: str = "auto",
        neighbours: int = 64,
        smoothing_length: float | None = None,
    ) -> DensityCubeData:
        assert self._reader.simulation_attributes is not None
        
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        # Data access for the galaxies/haloes
        group_data = self._collections[group_type]._data["properties"]
        assert isinstance(group_data, Group)
        
        # Centre of the cube, found my taking the mean of the centres of mass of each galaxy/halo in the group
        coms_dataset = group_data["core/com_pos_baryon"]
        assert isinstance(coms_dataset, Dataset)
        coms = coms_dataset[group_indices]
        mean_com = np.mean(coms_dataset, axis=0)
        
        if alignment is not None:
            if alignment == "angular_momentum":
                dataset = group_data["core/L_baryon"]
                assert isinstance(dataset, Dataset)
                total_L = np.sum(dataset[group_indices][:], axis=0) # L_star? L_gas?
            else:
                raise ValueError(f"Unknown alignment option: {alignment}")
        
        if ptype not in {"gas", "star"}:
            raise ValueError("ptype must be 'gas' or 'star'.")
        if smoothing not in {"auto", "provided", "nearest_neighbour", "fixed"}:
            raise ValueError(
                "smoothing must be 'auto', 'provided', 'nearest_neighbour', or 'fixed'."
            )
        if smoothing == "fixed" and (smoothing_length is None or smoothing_length <= 0):
            raise ValueError("A positive smoothing_length is required when smoothing='fixed'.")
        if smoothing == "provided" and ptype == "star":
            raise ValueError("Star particles do not provide SPH smoothing lengths.")

        use_provided = smoothing == "provided" or (smoothing == "auto" and ptype == "gas")
        gal_reqs = {
            ptype: ["mass", "pos"] + (["smoothing_length"] if use_provided else [])
        }
        gal_data = self._extract_groups(group_type=group_type, group_indices=group_indices, ptype_columns=gal_reqs)

        particles = _build_particle_stores(subset_data=gal_data, internals=self._internals)
        particle_data = particles[ptype]
        positions = particle_data["pos"]
        if use_provided:
            particle_smoothing = particle_data["smoothing_length"]
        elif smoothing == "fixed":
            particle_smoothing = np.full(len(positions), smoothing_length, dtype=np.float64)
        else:
            particle_smoothing = DensityCubeData.estimate_nearest_neighbour_smoothing_lengths(
                positions - mean_com,
                neighbours=neighbours,
            )

        return DensityCubeData(
            pos=positions,
            mass=particle_data["mass"],
            property=particle_data["mass"],
            smoothing_length=particle_smoothing,
            com=mean_com,
            angular_momentum=total_L if alignment == "angular_momentum" else None,
            config=self._config,
        )
        
    def get_num_particles(self, ptype: str, group_index: int):
        assert self._catalogue.galaxies is not None, "Catalogue is not set."
        return len(self._catalogue.galaxies.get_particle_indices(ptype, group_index))


def build_cube_analyser(
    catalogue: OctaviusCatalogue,
    config: OctaviusConfig,
    snapshot_path: Path | None = None,
) -> CubeAnalyser:
    """
    Constructs a CubeAnalyser object, from which one can build datacubes.
    
    `get_useful_density_cube_data`

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
