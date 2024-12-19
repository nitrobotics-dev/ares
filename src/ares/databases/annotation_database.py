from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pymongo import MongoClient

TEST_ANNOTATION_DB_PATH = "mongodb://localhost:27017/"


class AnnotationDatabase:
    def __init__(self, connection_string: str = TEST_ANNOTATION_DB_PATH):
        self.client = MongoClient(connection_string)
        self.db = self.client.video_annotations

        # Just two collections - videos and annotations
        self.videos = self.db.videos
        self.annotations = self.db.annotations
