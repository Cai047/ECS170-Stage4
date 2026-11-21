"""Save and load experiment summaries using the instructor's Result interface."""
import json
from pathlib import Path
from local_code.base_class.result import result

class Result_Saver(result):
    def _path(self):
        return Path(self.result_destination_folder_path) / self.result_destination_file_name

    def save(self):
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')

    def load(self):
        self.data = json.loads(self._path().read_text(encoding='utf-8'))
        return self.data
