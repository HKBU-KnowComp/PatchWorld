"""Shared configuration for the PatchWorld package."""

from typing import Dict

DEFAULT_SERVER_BASE: Dict[str, str] = {
    "alfworld": "http://127.0.0.1:36001",
    "sciworld": "http://127.0.0.1:36002",
    "babyai": "http://127.0.0.1:36003",
    "maze": "http://127.0.0.1:36004/maze",
    "wordle": "http://127.0.0.1:36004/wordle",
    "textcraft": "http://127.0.0.1:36005",
    "webshop": "http://127.0.0.1:36006",
    "webarena": "http://127.0.0.1:36007",
    "weather": "http://127.0.0.1:36008",
    "todo": "http://127.0.0.1:36009",
    "movie": "http://127.0.0.1:36010",
    "sheet": "http://127.0.0.1:36011",
    "academia": "http://127.0.0.1:36012",
    "searchqa": "http://127.0.0.1:36013",
    "sqlgym": "http://127.0.0.1:36014",
}

ENV_TASK_MAP: Dict[str, str] = {
    "alfworld": "AlfWorldTask",
    "sciworld": "SciworldTask",
    "babyai": "BabyAITask",
    "maze": "MazeTask",
    "wordle": "WordleTask",
    "textcraft": "TextCraftTask",
    "webshop": "WebshopTask",
    "webarena": "WebarenaTask",
    "weather": "WeatherTask",
    "todo": "TodoTask",
    "movie": "MovieTask",
    "sheet": "SheetTask",
    "academia": "AcademiaTask",
    "searchqa": "SearchQATask",
    "sqlgym": "SqlGymTask",
}

