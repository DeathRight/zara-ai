import datetime
import json
import random
import time


def get_time_info():
    """
    Gets the current time, timezone, and daylight savings time status

    Returns:
        A JSON string containing the current time, timezone, and daylight savings time status
        ```json
        {
            "time": "2019-03-10T15:00:00.000000",
            "timezone": "Central Daylight Time",
            "is_dst": true
        }
        ```
    """
    current_time = datetime.datetime.now()
    time_info = {}
    time_info["time"] = current_time.isoformat()
    time_info["timezone"] = time.tzname[0] if not time.daylight else time.tzname[1]
    time_info["is_dst"] = time.daylight and time.localtime().tm_isdst > 0

    return json.dumps(time_info)


def roll_dice(
    num_sides=20, num_dice=1, modifier=0, advantage=False, disadvantage=False
):
    """
    Rolls a number of dice with a given number of sides, optionally with a modifier and/or advantage/disadvantage

    Arguments:
        num_sides: The number of sides on each die
        num_dice: The number of dice to roll (default 1)
        modifier: The modifier to add to the roll (default 0)
        advantage: Whether to roll with advantage (default False)
        disadvantage: Whether to roll with disadvantage (default False)
    """
    if advantage and disadvantage:
        raise ValueError("Can't roll with both advantage and disadvantage.")

    # Ensure at least two dice are rolled if advantage or disadvantage is specified
    if advantage or disadvantage:
        num_dice = max(2, num_dice)

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]

    if advantage:
        roll_result = max(rolls)
    elif disadvantage:
        roll_result = min(rolls)
    else:
        roll_result = sum(rolls)

    return {"result": roll_result + modifier, "rolls": rolls}


# Usage:
print(roll_dice(advantage=True))
