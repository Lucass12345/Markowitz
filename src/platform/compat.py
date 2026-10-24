"""Compatibility URLs now dispatch workers; no HTTP training/optimization executor."""
from src.platform.jobs import JobService
from src.platform.operations import submit


class DurableManager:
    def submit(self,payload):return submit('optimization',payload)
    def get(self,identity):return JobService().get(identity)


def optimization_request(payload,**ignored):
    job=submit('optimization',payload)
    return {'run_id':None,'status':job['status'],'message':'Queued; query /jobs/'+job['id'],
            'result':{'job_id':job['id']},'mlflow_run_id':None,'mlflow_error':None}
