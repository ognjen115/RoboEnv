"""The lab cell's rigs (pure data, shared by task discovery and the runtime backend).

A rig is one combination of grippers, arm spacing and scene cameras; tasks with the same combination
share a rig and its device catalog. A new lab-sim task needs an entry here.
"""

RIGS = {
    "lab-xarm7-drawer": ["open_drawer", "close_drawer"],
    "lab-xarm7-cup": ["pour_from_cup", "handover_cup"],
    "lab-xarm7-tray": ["place_in_drawer", "stow_in_drawer"],
    "lab-xarm7-bin": ["clear_bin"],
}


def rig_of(task):
    for rig, tasks in RIGS.items():
        if task in tasks:
            return rig
    raise ValueError(f"no lab rig declares task {task!r}")
