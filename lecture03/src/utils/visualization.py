import numpy as np
import matplotlib.pyplot as plt
import pylab
from typing import Optional

def showSbs(a1:np.ndarray, a2:np.ndarray, stats : bool = False, bottom : str = "NN output", top : str = "Reference", title : Optional[str] = None):
    c = []

    for i in range(3):
        b = np.flipud( np.concatenate((a2[i],a1[i]),axis=1).transpose())
        min_value, mean_value, max_value = np.min(b), np.mean(b), np.max(b)

        if stats:
            print("stats : min - {}, mean - {}, max - {}".format(min_value, mean_value, max_value))
        
        b -= min_value
        b /= (max_value - min_value)
        c.append(b)
    
    fig, axes = pylab.subplots(1, 1, figsize=(16, 5))
    axes.set_xticks([])
    axes.set_yticks([])

    im = axes.imshow(np.concatenate(c,axis=1), origin='upper', cmap='magma')

    pylab.colorbar(im)
    pylab.xlabel('p, ux, uy')
    pylab.ylabel('%s %s'%(bottom,top))
    
    if title is not None: 
        pylab.title(title)