from celery import shared_task, Task
from celery.contrib.abortable import AbortableTask
from flask import current_app
from time import sleep

@shared_task(bind=True, base=AbortableTask)
def generate_from_text_picture(self, text, picture):

    for i in range(10):
        print(i)
        sleep(1)
        if self.is_aborted():
            return 'TASK STOPPED!'
    return f'DONE! {text}'

@shared_task(bind=True, base=AbortableTask)
def generate_from_text(self, text):

    for i in range(10):
        print(i)
        sleep(1)
        if self.is_aborted():
            return 'TASK STOPPED!'
    return f'DONE! {text}'