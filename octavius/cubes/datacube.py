
from typing import TYPE_CHECKING

# default libraries
from pathlib import Path
from dataclasses import dataclass

# other packages
import numpy as np


from ..log import get_logger, configure_logger

INTERNALS_FILEPATH = Path(__file__).parent.parent / "internals.yaml"

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import colors
# from https://github.com/brkent/datacube-display
CDICT_BKCMAP={
    'red'  :  ((0., 0., 0.), (0.25,0.,0.), (0.5,1.,1.), (0.75,1.0,1.0), (1., 1., 1.)),
    'green':  ((0., 0., 0.), (0.25,0.,0.), (0.5,0.,0.), (0.75,1.0,1.0), (1., 1., 1.)),
    'blue' :  ((0., 0., 0.), (0.25,1.,1.), (0.5,0.,0.), (0.75,0.0,0.0), (1., 1., 1.))
}
BKCMAP = colors.LinearSegmentedColormap('bkcmap', CDICT_BKCMAP, 1024)



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
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        
        # Imports for vispy rendering. TODO: Figure out how to make these optional / only included in dependencies of the user wants to use datacubes / this specific renderer
        from vispy import app, scene
        from vispy.color import Colormap
        from vispy.scene.visuals import Volume, XYZAxis
        import matplotlib as mpl
        from itertools import cycle
        
        if colormap is None:
            cmap = BKCMAP
        else:
            cmap = mpl.colormaps[colormap]

        data = np.asarray(self.data, dtype=np.float64)

        data_min = data.min() if vmin is None else vmin
        data_max = data.max() if vmax is None else vmax
        if data_min >= data_max:
            raise ValueError("vmin must be smaller than vmax.")
        
        
        # Wrangle annoying data values for logarithmic normalization
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
        
        log_vmin = data_max * 1e-3 # This is magic; Find a good way to set this. Was imagining a slider but idealy that'd come as part of a whole GUI
        
        # ColorBarWidget requires a VisPy Colormap. Sampling the Matplotlib
        # colormap preserves the same colors used for the volume textures.
        
        # Sample a VisPy colormap from the existing MPL colormap, to fit type requirements of the ColorBarWidget 
        vispy_cmap = Colormap(cmap(np.linspace(0.0, 1.0, 1024)))

        data_for_lin_norm = data
        lin_normalization = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
        
        data_for_log_norm = np.maximum(data, log_vmin)
        log_normalization = mpl.colors.LogNorm(vmin=log_vmin, vmax=data_max)
        
        normalized_lin_data = lin_normalization(data_for_lin_norm.reshape(-1)).reshape(data.shape)
        normalized_log_data = log_normalization(data_for_log_norm.reshape(-1)).reshape(data.shape)
        
        lin_vol = np.asarray(normalized_lin_data, dtype=np.float32)
        log_vol = np.asarray(normalized_log_data, dtype=np.float32)
    
        # Create a VisPy scene and add the volume visuals    
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        canvas.measure_fps()
        view = canvas.central_widget.add_view()

        lin_volume = Volume(
            lin_vol,
            parent=view.scene,
            method='mip',
            cmap=vispy_cmap,
            clim=(0.0, 1.0),
        )
        log_volume = Volume(
            log_vol,
            parent=view.scene,
            method='mip',
            cmap=vispy_cmap,
            clim=(0.0, 1.0),
        )
        log_volume.visible = False
        normalization = 'linear'
        
        volume_center = (np.array(vol_data.shape) / 2.0)[::-1]
        lin_volume.transform = 
        
        XYZAxis(pos=self., parent=view.scene, width=10)

        cam1 = scene.cameras.TurntableCamera(parent=view.scene, fov=60, name='Turntable')
        cam2 = scene.cameras.ArcballCamera(parent=view.scene, fov=60, name='Arcball')
        view.camera = cam1

        methods = cycle(['mip', 'attenuated_mip', 'translucent', 'additive', 'average', 'iso'])
        cameras = cycle([cam1, cam2])
        
        grid = canvas.central_widget.add_grid()
        cbar = scene.ColorBarWidget(cmap=vispy_cmap, orientation='right', clim=(data_min, data_max))
        grid.add_widget(cbar)
        cbar.visible = False        

        canvas_events = getattr(canvas, 'events')

        @canvas_events.key_press.connect
        def on_key_press(event):
            if event.text == '1':
                method = next(methods)
                lin_volume.method = method
                log_volume.method = method
                print("Render method: %s" % method)
            elif event.text == '2':
                view.camera = next(cameras)
                print(view.camera.name + ' camera')
            elif event.text == '3':
                nonlocal normalization
                normalization = {'linear': 'logarithmic', 'logarithmic': 'linear'}[normalization]
                lin_volume.visible = normalization == 'linear'
                log_volume.visible = normalization == 'logarithmic'
                print(f"{normalization.capitalize()} normalization")
            elif event.text == '4':
                cbar.visible = not cbar.visible

        app.run()
