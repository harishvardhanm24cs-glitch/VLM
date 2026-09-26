import asyncio
from unittest.mock import MagicMock
from VLM_Surveillance_Project.backend.unified_server import state, update_alarm_state
import VLM_Surveillance_Project.backend.unified_server as backend


def setup_mock():
    backend.esp32_client = MagicMock()

    asyncio.create_task = MagicMock()


def test_scenario(name, tracks, expected_alarm):

    backend.esp32_client.alarm_on.reset_mock()
    backend.esp32_client.alarm_off.reset_mock()
    asyncio.create_task.reset_mock()

    state.tracks = tracks
    update_alarm_state()

    if expected_alarm:
        assert state.esp32_alarm_active == True
    else:
        assert state.esp32_alarm_active == False


def run_tests():
    setup_mock()

    state.esp32_alarm_active = False

    test_scenario("TEST 1: Person", {"1": {"object_type": "person"}}, True)

    state.esp32_alarm_active = False
    test_scenario(
        "TEST 2: Military Vehicle",
        {"2": {"object_type": "truck", "vehicle_class": "MILITARY"}},
        False,
    )

    test_scenario(
        "TEST 3: Normal Vehicle",
        {"3": {"object_type": "car", "vehicle_class": "NORMAL"}},
        True,
    )

    state.esp32_alarm_active = False
    test_scenario(
        "TEST 4: Unknown Vehicle",
        {"4": {"object_type": "bus", "vehicle_class": "UNCERTAIN"}},
        True,
    )

    state.esp32_alarm_active = False
    test_scenario(
        "TEST 5: Military + Person",
        {
            "1": {"object_type": "person"},
            "2": {"object_type": "truck", "vehicle_class": "MILITARY"},
        },
        True,
    )

    state.esp32_alarm_active = False
    test_scenario(
        "TEST 6: Military + Normal",
        {
            "3": {"object_type": "car", "vehicle_class": "NORMAL"},
            "2": {"object_type": "truck", "vehicle_class": "MILITARY"},
        },
        True,
    )

    state.esp32_alarm_active = True
    test_scenario("TEST 7: Empty Scene", {}, False)


if __name__ == "__main__":
    run_tests()
