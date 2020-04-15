from abc import abstractmethod


class Dataset:

    @property
    @abstractmethod
    def directory(self) -> str:
        pass

    @property
    @abstractmethod
    def activities(self) -> list:
        pass

    @property
    @abstractmethod
    def files(self) -> list:
        pass

    @property
    @abstractmethod
    def extensions(self) -> list:
        pass

    @property
    @abstractmethod
    def extensions_activities(self) -> list:
        pass

    @property
    @abstractmethod
    def sensors(self) -> list:
        pass
