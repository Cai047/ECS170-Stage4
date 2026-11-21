"""Retain the official test partition and use a stratified training/validation split."""
from local_code.base_class.setting import setting

class Setting_Train_Validation_Test(setting):
    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        self.method.data = loaded_data
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None
