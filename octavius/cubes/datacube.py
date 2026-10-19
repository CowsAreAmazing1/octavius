
from dataclasses import dataclass

import numpy as np

@dataclass
class Axis:
    """One axis of a datacube. Allows clean specification of the axis and its properties."""
    
    name: str
    unit: str
    coordinates: np.ndarray
    
    
@dataclass
class DataCube:
    """A datacube with 3 axes"""
    
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


    def animate_slice(self, filepath: str, fps: int = 30):
        # TODO: Get info about units here. Add useful axis labels in general.
        
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
            
