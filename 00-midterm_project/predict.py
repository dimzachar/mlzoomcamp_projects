import bentoml

from bentoml.io import JSON
from bentoml.io import NumpyNdarray


from pydantic import BaseModel

class UserProfile(BaseModel):
    relative_compactness: float
    surface_area: float
    wall_area: float
    roof_area: float
    overall_height: float
    orientation: int
    glazing_area: float
    glazing_area_distribution: int

model_ref = bentoml.xgboost.get("energy_efficiency_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("energy_efficiency_regressor", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=UserProfile), output=JSON())
async def classify(user_profile):
    application_data = user_profile.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
   

    result = prediction[0]
    print(result)
    print(result[0]+result[1])
    if (result[0]+result[1]) > 65:
        return {"total_load" : "high"}
    elif (result[0]+result[1]) > 29:
        return {"total_load" : "average"}
    else:
        return {"total_load" : "low"}
    
    return {"[heating_load, cooling_load]" : result}
