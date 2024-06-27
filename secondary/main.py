from fastapi import FastAPI, BackgroundTasks


app = FastAPI(
    title="II. Inference Server"
)


def async_infernce():
    pass


@app.route("/summarize")
def summarize(
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(async_infernce)

    return "Success"