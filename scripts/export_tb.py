from pathlib import Path
import csv
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

logdir = Path("/home/rog-server/esrl/runs/train/exp158")
outpath = logdir / "all_scalars_wide.csv"

event_files = sorted(logdir.glob("events.out.tfevents.*"))
if not event_files:
    raise SystemExit(f"No event files found in {logdir}")
event_path = event_files[-1]

ea = event_accumulator.EventAccumulator(
    str(event_path),
    size_guidance={event_accumulator.SCALARS: 0},
)
ea.Reload()

tags = ea.Tags().get("scalars", [])
# step -> {tag: value}
by_step = defaultdict(dict)

for tag in tags:
    for r in ea.Scalars(tag):
        by_step[int(r.step)][tag] = float(r.value)

steps = sorted(by_step.keys())
tags = sorted(tags)

with outpath.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step"] + tags)
    for step in steps:
        row = [step] + [by_step[step].get(tag, "") for tag in tags]
        w.writerow(row)

print(f"Wrote {outpath} with {len(steps)} steps and {len(tags)} tags")