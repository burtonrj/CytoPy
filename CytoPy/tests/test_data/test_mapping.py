from ...data.mapping import ChannelMap
import pytest


@pytest.fixture()
def example_channel_map():
    return ChannelMap(channel="test channel", marker="test marker")


@pytest.mark.parametrize("channel,marker,expected",
                         [("test channel", "test marker", True),
                          ("test marker", "test channel", False),
                          ("invalid", "test marker", False)])
def test_matching_pair(example_channel_map, channel, marker, expected):
    cm = example_channel_map
    assert cm.check_matched_pair(channel=channel, marker=marker) == expected


def test_to_dict(example_channel_map):
    d = example_channel_map.to_dict()
    assert isinstance(d, dict)
    assert d.get("channel") == "test channel"
    assert d.get("marker") == "test marker"
