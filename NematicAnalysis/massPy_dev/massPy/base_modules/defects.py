from sys import version_info as py_version
from . import numdiff
import numpy as np

if py_version < (3, 10):
    from itertools import tee
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
else:
    from itertools import pairwise


def get_charge(vx, vy, LX, LY):
    """
    Compute the charge array associated with a two components order-parameter field.
    Defects then show up as small regions of non-zero charge (typically 2x2). \n
    Args:
        vx, vy: The components of the order-parameter field.
        LX, LY: Size of domain.
    Returns:
        Charge field with shape (LX, LY).
    """
    # compute angle 
    def wang(a, b):
        """ --== Infamous chinese function (馬子目水) ==-- """
        ang = np.arctan2(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1])
        return np.where( np.abs(ang) > .5*np.pi,    # if
                         ang - np.sign(ang)*np.pi,  # then
                         ang                        # else
                        )
    
    # stack vector field along first axis and reshape
    v = np.stack([vx, vy]).reshape(2, LX, LY)
    
    # instantiate charge array
    w = np.zeros((LX, LY))
    
    # compute the winding number around every point:
    #       ( 1, 1) ( 1, 0) ( 1, -1)
    #       ( 0, 1)  point  ( 0, -1)
    #       (-1, 1) (-1, 0) (-1, -1)
    # --------------------------------------------
    for s0, s1 in pairwise([ (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0) ]):
        v0 = np.roll(v, shift=s0, axis=(1,2))
        v1 = np.roll(v, shift=s1, axis=(1,2))
        w += wang(v0, v1)

    return w / (2*np.pi)


def get_diffusive_charge_density(vx, vy, LX, LY):
    """ whatever """
    return numdiff.jacobian(vx.reshape(LX, LY), vy.reshape(LX, LY))


def get_defects(vx, vy, LX, LY, threshold):
    """
    Returns list of defects from the order-parameter field. \n
    Args:
        vx, vy: The components of the order-parameter field.
        LX, LY: Size of domain.
        threshold: For defect detection in charge array.
    Returns:
        List of the form [ {'pos': (x, y), 'charge': w} ].
    """
    # get charge charge array w
    w = get_charge(vx, vy, LX, LY)
    
    # defects show up as 2x2 regions in the charge array w and must be
    # collapsed to a single point by taking the average position of
    # neighbouring points with the same charge (by breadth-first search).
        
    def collapse(root, s):
        # breadth-first search (BFS)
        def recursive_bfs(queue, detected):
            # if queue is empty, then terminate
            if not queue: return np.mean(detected, 0)
            # if queue is NOT empty, then dequeue
            i, j = queue.pop(0)
            # check every neighbour
            for nbr in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                ni = nbr[0] % LX    # periodic boundaries
                nj = nbr[1] % LY    #       --||--
                # same charge as root and not previously detected
                if s*w[ni, nj] > threshold and nbr not in detected:
                    detected.append(nbr)    # mark as detected
                    queue.append(nbr)       # enqueue it
                    w[ni, nj] = .0          # inform scope
            # call recursively
            return recursive_bfs(queue, detected)
        w[root] = .0    # inform the larger scope of detection
        x, y = recursive_bfs([root], [root])
        return x % LX, y % LY
    
    d = []
    
    for idx in np.ndindex(w.shape):
        if abs(w[idx]) > threshold:
            # charge sign
            s = np.sign(w[idx])
            # breadth-first search (BFS)
            x, y = collapse(idx, s)
            # add defect to list
            d.append({
                'charge': .5*s,
                'pos': [x, y]
            })
    return d
