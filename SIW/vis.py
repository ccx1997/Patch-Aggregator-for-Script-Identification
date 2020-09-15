# -*- coding: utf-8 -*-
import visdom
import numpy as np
import time

vis = visdom.Visdom()
assert vis.check_connection()
# vis.close()
for i in range(15):
    vis.line(X=np.array([i]), Y=np.array([i+2]), update='append',
            win='here', name='2', opts=dict(showlegend=True))
    time.sleep(1)


