from pydoc import cli #where is thi sfrom and to do what?
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union #explain this
from enum import Enum #explain this

class Applicant(BaseModel):
    possessed_skills: Union[str, None] = None
    salary_mean_offer: Union[float, None] = None
    years_of_experience: Union[float, None] = None

class Description(BaseModel):
    description: Union[str, None] = Field(None, title = "Your job post description", max_length = 4000)
    mean_salary: Union[float, None] = 0

class ModelName(str, Enum):
    description_to_job = "job_model"
    skills_needed = "skills_model"
    salary_offer_estimator = "hr_model"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/models/{model_name}") ### can we use fastapi.HTTPException for missing models?
def get_model(model_name : ModelName): ###here https://fastapi.tiangolo.com/tutorial/path-params/ they type the param as a ModelName object, but then it breaks on nonexistent calls? Is that better or having some default page?
    if model_name == ModelName.description_to_job:
        return {"model_name": model_name, "message": "This can give the job level"}
    if model_name == ModelName.skills_needed:
        return {"model_name": model_name, "message": "These are what you need for that job"}
    if model_name == ModelName.salary_offer_estimator:
        return {"model_name": model_name, "message": "This is the expected salary for that description"}
    return {"model_name": model_name, "message": "This isn't a defined model"}

client_description_request = Description()
#async?
@app.post("/models/job_model")
def take_job_desc(desc : Description):
    ###Here is where the post input get's turned into our model's output
    # user_input = [stream]
    # res = model(user_input)
    # interpretation = interpret_results(res)
    # return interpretation
    # return {"job level" : model.predict(desc)}
    client_description_request.description = desc
    return dict(job_level= desc)

@app.get("/models/job_model/job_results")
def present_job_results():
    return {"job level" : f"{client_description_request.description}, I was added in the present function"}
