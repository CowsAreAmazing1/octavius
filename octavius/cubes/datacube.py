
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.loader import OctaviusCatalogue, GroupCollection
    from ..data_management import SnapshotReader
    
# default libraries
from pathlib import Path
from dataclasses import dataclass

# other packages
import numpy as np
import h5py
import numba
from scipy.spatial import KDTree
from scipy.sparse import csr_array

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


@dataclass
class Axis:
    """One axis of a datacube. Allows clean specification of the axis and its properties."""
    
    name: str
    unit: str
    coordinates: np.ndarray
    
    
@dataclass
class DataCube:
    """A generic datacube with 3 axes"""
    
    data: np.ndarray
    axes: tuple[Axis, Axis, Axis]
    quantity: str
    unit: str    
    
    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError("Data must be a 3D array. Got shape: {}".format(self.data.shape))
        
        if len(self.axes) != 3:
            raise ValueError("There must be exactly 3 axes. Got: {}".format(len(self.axes)))
        
        for dimension, axis in enumerate(self.axes):
            if len(axis.coordinates) != self.data.shape[dimension]:
                raise ValueError(
                    f"Length of axis '{axis.name}' coordinates ({len(axis.coordinates)}) "
                    f"does not match data shape along that axis ({self.data.shape[dimension]})."
                )
                
    @property
    def shape(self):
        return self.data.shape


    def animate_slice(self, filepath: str | Path, fps: int = 30):
        # TODO: Get info about units here. Add useful axis labels in general.
        
        filepath = Path(filepath) if isinstance(filepath, str) else filepath
        if filepath.is_file() and input(f"File {filepath} already exists. Overwrite? (y/n): ").lower() != "y":
            print("Aborting animation save.")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib import colors, cm
        
        # from https://github.com/brkent/datacube-display
        cdict_bkcmap={
            'red'  :  ((0., 0., 0.), (0.25,0.,0.), (0.5,1.,1.), (0.75,1.0,1.0),  (1., 1., 1.)),
            'green':  ((0., 0., 0.), (0.25,0.,0.), (0.5,0.,0.), (0.75,1.0,1.0),  (1., 1., 1.)),
            'blue' :  ((0., 0., 0.), (0.25,1.,1.), (0.5,0.,0.), (0.75,0.0,0.0),  (1., 1., 1.))
        }

        bkcmap = colors.LinearSegmentedColormap('bkcmap', cdict_bkcmap, 1024) # type: ignore

        m, M = self.data.min(), self.data.max()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax1.imshow(self.data[0, :, :], cmap=bkcmap, vmin=m, vmax=M)
        im2 = ax2.imshow(self.data[:, 0, :], cmap=bkcmap, vmin=m, vmax=M)
        im3 = ax3.imshow(self.data[:, :, 0], cmap=bkcmap, vmin=m, vmax=M)
        ax1.set_title("Slice: x = {self.axes[0].coordinates[0]:.2f} {self.axes[0].unit}")
        
        # fig.colorbar(im1, ax=ax1)
        # fig.colorbar(im2, ax=ax2)
        # fig.colorbar(im3, ax=ax3)
        
        def update(frame):
            im1.set_array(self.data[frame, :, :])
            im2.set_array(self.data[:, frame, :])
            im3.set_array(self.data[:, :, frame])
            ax1.set_title(f"Slice {frame}")
            
            return [im1, im2, im3]
        
        ani = animation.FuncAnimation(
            fig, update, frames=self.shape[0], interval=100, blit=True
        )

        # Note: Requires the 'pillow' library installed (pip install pillow)
        writer = animation.PillowWriter(fps=fps)
        ani.save(filepath, writer=writer)
   
   
   
            
class CubeAnalyser(OctaviusAnalyser):
    """Start of a class to handle building datacubes"""
    
    # TODO: These functions are obviously not very general currently
    def get_useful_density_cube_data(
        self,
        group_indices: list[int] | np.ndarray,
        alignment: str | None = None,
    ):
        assert self._reader.simulation_attributes is not None
        
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
        group_type = "galaxies"
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        # Data access for the galaxies/haloes
        group_data = self._collections[group_type]._data["properties"]
        
        # Centre of the cube
        mean_com = np.mean(group_data["core/com_pos_baryon"][group_indices], axis=0)
        
        if alignment is not None:
            if alignment == "angular_momentum":
                mean_L = np.mean(group_data["core/L_star"][group_indices], axis=0)
                mean_L /= np.linalg.norm(mean_L)
            else:
                raise ValueError(f"Unknown alignment option: {alignment}")
        
        

        gal_reqs = {
            'star': ['mass', 'pos'],
            'gas':  ['mass', 'pos'] # 'smoothing_length'
        }
        gal_data = self._extract_groups(group_type=group_type, group_indices=group_indices, ptype_columns=gal_reqs)

        particles = _build_particle_stores(subset_data=gal_data, internals=self._internals)
        galaxies = _build_group_store(group_type="galaxies", group_indices=group_indices, subset_data=gal_data, internals=self._internals)

        # sim_data = SimulationData(
        #     simulation=self._reader.simulation_attributes,
        #     constants=self._constants,
        #     particles=particles,
        #     groups=groups,
        # )
        
        stars = particles["star"]
        
        from ..cubes.density_cube import DensityCubeData
        
        return DensityCubeData(
            pos=stars["pos"],
            mass=stars["mass"],
            com=mean_com,
            L=mean_L if alignment == "angular_momentum" else None,
            # centre=simulation_data.groups["galaxies"]["core/com_pos_baryon"]
        )


def build_cube_analyser(
    catalogue: OctaviusCatalogue,
    config: OctaviusConfig,
    snapshot_path: Path | None = None,
) -> CubeAnalyser:
    """
    Constructs a CubeAnalyser object, from which one can build datacubes.
    
    `get_useful_density_cube_data`
    `wrangle_density_data`

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
