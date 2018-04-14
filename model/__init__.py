"""
Define the CNN model used by Res-CNN, including the PRJ, FBP, RFN subnets
"""

# we will always use these APIs
from .subnet.fbp import fbp_subnet as fbp_network
from .subnet.prj_est_impl import slice_concat
