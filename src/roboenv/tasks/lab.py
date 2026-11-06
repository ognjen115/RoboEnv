"""The lab cell's rigs (pure data, shared by task discovery and the runtime backend).

A rig is one combination of grippers, arm spacing and scene cameras; tasks with the same combination
share a rig and its device catalog. A new lab-sim task needs an entry here.
"""

# the standard cell (arms 0.72 m apart, the right fixed camera): one rig per gripper pair
RIGS = {
    "lab-xarm7-jaw-jaw": ["open_drawer", "close_drawer", "pour_from_cup", "handover_cup"],
    "lab-xarm7-jaw-vacuum": ["place_in_drawer", "stow_in_drawer"],
    "lab-xarm7-vacuum-jaw": ["clear_bin"],
}


def rig_of(task):
    for rig, tasks in RIGS.items():
        if task in tasks:
            return rig
    raise ValueError(f"no lab rig declares task {task!r}")
