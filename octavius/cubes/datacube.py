
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.loader import OctaviusCatalogue
    
# default libraries
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# other packages
import numpy as np

from ..utils.dynamic_analyser import (
    OctaviusAnalyser,
    _build_particle_stores,
)
from ..data_management import (
    OctaviusConfig,
    OctaviusConstants,
    build_reader,
)
from ..data_management.pipeline_management import load_internals, Internals, PipelineStage

from ..log import get_logger, configure_logger

INTERNALS_FILEPATH = Path(__file__).parent.parent / "internals.yaml"


from matplotlib import colors
# from https://github.com/brkent/datacube-display
CDICT_BKCMAP={
    'red'  :  ((0., 0., 0.), (0.25,0.,0.), (0.5,1.,1.), (0.75,1.0,1.0),  (1., 1., 1.)),
    'green':  ((0., 0., 0.), (0.25,0.,0.), (0.5,0.,0.), (0.75,1.0,1.0),  (1., 1., 1.)),
    'blue' :  ((0., 0., 0.), (0.25,1.,1.), (0.5,0.,0.), (0.75,0.0,0.0),  (1., 1., 1.))
}
BKCMAP = colors.LinearSegmentedColormap('bkcmap', CDICT_BKCMAP, 1024) # type: ignore



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
        


        m, M = self.data.min(), self.data.max()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = ax1.imshow(self.data[0, :, :], cmap=BKCMAP, vmin=m, vmax=M)
        im2 = ax2.imshow(self.data[:, 0, :], cmap=BKCMAP, vmin=m, vmax=M)
        im3 = ax3.imshow(self.data[:, :, 0], cmap=BKCMAP, vmin=m, vmax=M)
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
        
    def render_vispy(
        self,
        colormap: str | None = None,
        norm: str = "linear",
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        
        from vispy import app, scene
        import matplotlib as mpl
        from itertools import cycle
        
        if colormap is None:
            cmap = BKCMAP
        else:
            cmap = mpl.colormaps[colormap]

        if norm not in {"linear", "log"}:
            raise ValueError("norm must be either 'linear' or 'log'.")

        data = np.asarray(self.data, dtype=np.float64)
        # if not np.all(np.isfinite(data)):
        #     raise ValueError("Data must contain only finite values for rendering.")

        data_min = data.min() if vmin is None else vmin
        data_max = data.max() if vmax is None else vmax
        if data_min >= data_max:
            raise ValueError("vmin must be smaller than vmax.")

        if norm == "log":
            positive_data = data[data > 0] # LogNorm complains if vmin <= 0, so set the colour floor to the smallest data value > 0
            if positive_data.size == 0:
                raise ValueError("Logarithmic normalization requires at least one positive value.")
            log_vmin = positive_data.min() if vmin is None else vmin
            if log_vmin <= 0:
                raise ValueError("vmin must be positive for logarithmic normalization.")
            if vmax is None:
                data_max = positive_data.max()
            if data_max <= log_vmin:
                raise ValueError("vmax must be greater than vmin for logarithmic normalization.")
            data_for_norm = np.maximum(data, log_vmin)
            normalization = mpl.colors.LogNorm(vmin=log_vmin, vmax=data_max)
        else:
            normalization = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
            data_for_norm = data

        normalized_data = normalization(data_for_norm.reshape(-1)).reshape(data.shape)
        vol = cmap(normalized_data)[..., :3].astype(np.float32)
        
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        canvas.measure_fps()

        view = canvas.central_widget.add_view()
        volume = scene.visuals.Volume(vol, parent=view.scene, method='mip')

        cam1 = scene.cameras.TurntableCamera(parent=view.scene, fov=60, name='Turntable')
        cam2 = scene.cameras.ArcballCamera(parent=view.scene, fov=60, name='Arcball')
        view.camera = cam1

        methods = cycle(['mip', 'attenuated_mip', 'translucent', 'additive',
                        'average', 'iso'])

        @canvas.events.key_press.connect
        def on_key_press(event):
            if event.text == '1':
                method = next(methods)
                volume.method = method
                print("Render method: %s" % method)
            elif event.text == '2':
                cam_toggle = {cam1: cam2, cam2: cam1}
                view.camera = cam_toggle.get(view.camera, cam1)
                print(view.camera.name + ' camera')

        app.run()


            
class CubeAnalyser(OctaviusAnalyser):
    """Start of a class to handle building datacubes"""
    
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
    ):
        assert self._reader.simulation_attributes is not None
        
        group_indices = np.sort(np.asarray(group_indices, dtype=np.int64))
        
        if group_type not in self._collections:
            raise KeyError(f"'{group_type}' is not present in the catalogue.")
        
        # Data access for the galaxies/haloes
        group_data = self._collections[group_type]._data["properties"]
        
        # Centre of the cube
        centres_of_mass = group_data["core/com_pos_baryon"][group_indices]
        mean_com = np.mean(centres_of_mass, axis=0)
        
        if alignment is not None:
            if alignment == "angular_momentum":
                total_L = np.sum(group_data["core/L_baryon"][group_indices], axis=0) # L_star? L_gas?
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
            from ..cubes.density_cube import DensityCubeData
            particle_smoothing = DensityCubeData.estimate_nearest_neighbour_smoothing_lengths(
                positions - mean_com,
                neighbours=neighbours,
            )

        from ..cubes.density_cube import DensityCubeData
        return DensityCubeData(
            pos=positions,
            mass=particle_data["mass"],
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
