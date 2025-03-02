
from .. import model_factory

from .geotr import LitGeoTr
from .geotr_template import LitGeoTrTemplate
from .geotr_template_large import LitGeoTrTemplateLarge
from .geotr_template_huge import LitGeoTrTemplateHuge

model_factory.register_model("geotr", LitGeoTr)
model_factory.register_model("geotr_template", LitGeoTrTemplate)
model_factory.register_model("geotr_template_large", LitGeoTrTemplateLarge)
model_factory.register_model("geotr_template_huge", LitGeoTrTemplateHuge)
