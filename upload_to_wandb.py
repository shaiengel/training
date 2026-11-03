import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {
    # "lr-1e-4": r"C:\portal\lr-output\lr-1e-4\lr-1e-4\runs\Nov01_22-02-13_da10499516db",
    "lr-3e-5": r"C:\portal\lr-output\lr-3e-5\lr-3e-5\Nov02_14-18-44_d8dd8d96054a",
    "lr-1e-5": r"C:\portal\lr-output\lr-1e-5\lr-1e-5\Nov02_15-17-38_d8dd8d96054a",
    "lr-5e-6": r"C:\portal\lr-output\lr-5e-6\lr-5e-6\Nov02_14-57-27_d8dd8d96054a",
}

for name, path in runs.items():
    ea = EventAccumulator(path)
    ea.Reload()

    run = wandb.init(project="learning-rate", name=name, reinit=True)

    all_steps = set()
    for tag in ea.Tags()["scalars"]:
        for event in ea.Scalars(tag):
            all_steps.add(event.step)

    for step in sorted(all_steps):
        data = {}
        for tag in ea.Tags()["scalars"]:
            for event in ea.Scalars(tag):
                if event.step == step:
                    data[tag] = event.value
        if data:
            run.log(data, step=step)

    run.finish()
    print(f"Uploaded {name}")
