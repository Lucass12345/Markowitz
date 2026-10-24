import os
from celery import Celery

app = Celery('markowitz',broker=os.getenv('CELERY_BROKER_URL','redis://127.0.0.1:6379/0'),include=['worker.tasks'])
app.conf.update(task_serializer='json',accept_content=['json'],result_serializer='json',
    task_acks_late=True,task_reject_on_worker_lost=True,worker_prefetch_multiplier=1,
    task_ignore_result=True,broker_connection_retry_on_startup=True,
    broker_transport_options={'visibility_timeout':90000,'socket_connect_timeout':3,'socket_timeout':3},
    task_default_queue='cpu',worker_max_tasks_per_child=20,worker_max_memory_per_child=int(os.getenv('PLATFORM_WORKER_MAX_MEMORY_KB','4000000')))
