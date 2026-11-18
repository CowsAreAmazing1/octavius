"""
Internal functions for building datacubes, using existing methods from the `OctaviusAnalyser` class.
Handles access to all data avaliable in the catalogue, snapshot and config.
"""
from octavius.data_management.snapshot_readers import SimbaReader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_management import OctaviusConfig
    from ..utils.loader import OctaviusCatalogue

# default packages
from pathlib import Path

# other packages
import numpy as np
from h5py._hl.dataset import Dataset
from h5py._hl.group import Group

# internal imports
from ..data_management import (
    OctaviusConstants,
    build_reader,
)
from ..data_management.pipeline_management import load_internals
from ..utils.dynamic_analyser import (
    OctaviusAnalyser,
    _build_particle_stores,
)
from .density_cube import DensityCubeData

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
                
    def splat_datasets(
        self
    ):
        """
        Prints a bunch of internal column? names used when reading snapshot data, specific to a gadget snapshot
        """
        print("===== READ RAW ========")
        print("dataset map:")
        for (k, v) in self._reader.dataset_map.items():
            print(f"    {k} : {v}")
        print("inverse ptype map:")
        for k, v in self._reader.inverse_ptype_map.items():
            print(f"    {k} : {v}")
    
    def read_simba_snapshot(
        self,
        group_type: str,
        group_indices: list[int] | np.ndarray,
        property_name: str,
        ptype: str,
    ):
        # i think this is simba-specific right now
        
        # Hack a smoothing length property into the underlying snapshot reader, for ptypes without a
        # proper "smoothing_length" ("SmoothingLength" ...) field. All ptypes (in a gadget snapshot)
        # have a somewhat equivalent AGS-Softening column
        if ptype != "gas":
            self._reader.dataset_map_overrides[(ptype, "smoothing_length")] = "AGS-Softening"
        
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
        # Center of the cube in physical coordinates
        group_data = self._collections[group_type]._data["properties"]
        assert isinstance(group_data, Group)
        com_pos_baryon = group_data["core/com_pos_baryon"]
        assert isinstance(com_pos_baryon, Dataset)
        coms = com_pos_baryon[group_indices]
        mean_com = np.mean(coms, axis=0)
        
        spatial_columns = {ptype: [
            "pos",
            "smoothing_length",
            property_name
        ]}
        spatial_subset_data = self._extract_groups(
            group_type=group_type,
            group_indices=group_indices,
            ptype_columns=spatial_columns,
        )
        
        particles = _build_particle_stores(subset_data=spatial_subset_data, internals=self._internals)
        spatial_data = particles[ptype]
        
        group_collection = self._catalogue.haloes if group_type == "galoes" else self._catalogue.galaxies
        assert group_collection is not None
        
        
        return DensityCubeData(
            pos=spatial_data["pos"],
            property=spatial_data[property_name],
            smoothing_length=spatial_data["smoothing_length"],
            com=mean_com,
            angular_momentum=None,
            config=self._config,
        )
        
    def particle_smoothing_column(self, ptype: str) -> str:
        if isinstance(self._reader, SimbaReader):
            if ptype == "gas": # The smoothing_length column already exists
                raise ValueError("This should really be checked first")
            return "AGS-Smoothing"
        raise NotImplementedError("Havent looked at snapshots from other simulations yet")
        
    # def get_spatial_cube_data(
    #     self,
    #     group_type: str,
    #     group_indices: list[int] | np.ndarray,
    #     property_name: str, 
    #     ptype: str =  "gas",
    #     alignment: str | None = None,
    # ):
    #     """
    #     Collects required data from the catalogue and snapshot to build a spatial datacube for the desired property, for the specified galaxies/haloes.
        
    #     Parameters
    #     ----------
    #     group_type: str
    #         The type of group to run on. Must be present in the catalogue.
    #     group_indices: list[int]
    #         The indices into the Octavius catalogue of the galaxies to run on.
    #     property_name: str
    #         The name of the property to build the datacube for.
    #         like `particle_specific/sfr` or `core/kappa_rot_gas`
    #     ptype: str
    #         The particle type to use. Must be present in the snapshot. Default: `gas`
    #     alignment: str | None
    #         The alignment method to use. TODO allow alignment with angular momentum. Im imagining that this will flatten the object into the xy plane.
    #         Also just allow a general alignment/rotation to be passed, if thats useful.

    #     Returns
    #     -------
    #     result: DensityCubeData
    #         A `DensityCubeData` dataclass from 
    #     """
    #     assert self._reader.simulation_attributes is not None
        
    #     group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
    #     if group_type not in self._collections:
    #         raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
    #     group_data = self._collections[group_type]._data['properties']
    #     assert isinstance(group_data, Group)
        
    #     # Center of the cube in physical coordinates
    #     com_pos_baryon = group_data["core/com_pos_baryon"]
    #     assert isinstance(com_pos_baryon, Dataset)
    #     coms = com_pos_baryon[group_indices]
    #     mean_com = np.mean(coms, axis=0)
        
    #     if alignment is not None:
    #         raise NotImplementedError("hold on")
        
    #     if ptype != "gas":
    #         # No idea how to handle things that arent gas particles yet. From what i can
    #         # see, gas particles are the only ones that have a smoothing length property,
    #         # so i guess the only ones that are SPHly modelled?
    #         raise NotImplementedError("Only gas particles are currently supported.")
        
    #     spatial_columns = { ptype : ["pos", "smoothing_length"] }
    #     spatial_subset_data = self._extract_groups(
    #         group_type=group_type,
    #         group_indices=group_indices,
    #         ptype_columns=spatial_columns,
    #     )    
    #     particles = _build_particle_stores(subset_data=spatial_subset_data, internals=self._internals)
    #     spatial_data = particles[ptype]
        
    #     property_data = self.get_manual_property_data(group_type=group_type, property_name=property_name)
        
    #     return DensityCubeData(
    #         pos=spatial_data['pos'],
    #         property=property_data,
    #         smoothing_length=spatial_data['smoothing_length'],
    #         com=mean_com,
    #         angular_momentum=None,
    #         config=self._config,   
    #     )
    
    # TODO: These functions are obviously not very general currently
    def get_useful_density_cube_data(
        self,
        group_type: str,
        group_indices: list[int] | np.ndarray,
        alignment: str | None = None,
        ptype: str = "gas",
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
        mean_com = np.mean(coms, axis=0)
        
        if alignment is not None:
            if alignment == "angular_momentum":
                dataset = group_data["core/L_baryon"]
                assert isinstance(dataset, Dataset)
                total_L = np.sum(dataset[group_indices][:], axis=0) # L_star? L_gas?
            else:
                raise ValueError(f"Unknown alignment option: {alignment}")
        
        if ptype not in {"gas", "star"}:
            raise ValueError("ptype must be 'gas' or 'star'.")

        gal_reqs = {ptype: ["mass", "pos", "smoothing_length"] }
        gal_data = self._extract_groups(group_type=group_type, group_indices=group_indices, ptype_columns=gal_reqs)

        particles = _build_particle_stores(subset_data=gal_data, internals=self._internals)
        particle_data = particles[ptype]
        
        return DensityCubeData(
            pos=particle_data["pos"],
            property=particle_data["mass"],
            smoothing_length=particle_data["smoothing_length"],
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
