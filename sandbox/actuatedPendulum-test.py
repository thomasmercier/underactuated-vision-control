import actuatedPendulum
import numpy as np

pend = actuatedPendulum.ControlledPendulum()
pend.learn()
pend.go()
