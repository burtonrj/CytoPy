import mongoengine

class ChannelMap(mongoengine.EmbeddedDocument):
    """
    Defines channel/marker mapping. Each document will contain a single value for channel and a single value for marker,
    these two values are treated as a pair within the panel.

    Parameters
    ----------
    channel: str
        name of channel (e.g. fluorochrome)
    marker: str
        name of marker (e.g. protein)
    """
    channel = mongoengine.StringField()
    marker = mongoengine.StringField()

    def check_matched_pair(self, channel: str, marker: str) -> bool:
        """
        Check a channel/marker pair for resemblance

        Parameters
        ----------
        channel: str
            channel to check
        marker: str
            marker to check

        Returns
        --------
        bool
            True if equal, else False
        """
        if self.channel == channel and self.marker == marker:
            return True
        return False

    def to_dict(self) -> dict:
        """
        Convert object to python dictionary

        Returns
        --------
        dict
        """
        return {'channel': self.channel, 'marker': self.marker}