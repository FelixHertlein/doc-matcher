from .. import model_factory

from .dewarpnet import LitDewarpNetWC
from .dewarpnet import LitDewarpNetBM
from .dewarpnet import LitDewarpNetJoint

model_factory.register_model("dewarpnet_wc", LitDewarpNetWC)
model_factory.register_model("dewarpnet_bm", LitDewarpNetBM)
model_factory.register_model("dewarpnet", LitDewarpNetJoint)
