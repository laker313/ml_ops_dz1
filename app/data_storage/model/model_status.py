from enum import Enum

class Learning_status(Enum):
    LEARNED = "learned"
    NOT_LEARNED = "not_learned"


def get_learning_status(status_str: str) -> Learning_status:
    """Преобразовать строку в Learning_status"""
    try:
        return Learning_status(status_str)
    except ValueError:
        raise ValueError(f"Invalid learning status: {status_str}")