from file import *
from pipeline import *
import datetime, os



class Log(File) :
    class HaltError(Exception):
        pass

    def __init__(self, start_stage: PipelineStage, target_stage: PipelineStage, log_folder_path: Path, start_folder_path: Path, target_folder_path: Path) -> None:
        super().__init__(os.path.join(log_folder_path, f"{start_stage.name}_to_{target_stage.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))
        self.start_stage: PipelineStage = start_stage
        self.target_stage: PipelineStage = target_stage
        self.start_stage_folder_path: Path = start_folder_path
        self.target_stage_folder_path: Path = target_folder_path
        self.text: List[str] = [self.name + f"\n {start_folder_path} -> {start_folder_path}" + 2 * "\n"]
        
        self.num_attempted: int = 0
        self.num_skipped: int = 0 
        self.num_successful: int = 0 
        self.num_errors: int = 0
        self.num_warned_successful: int = 0
        self.has_halted: bool = False


    def skip(self, input_file: File, output_file: File, reason: str = "") -> None:
        self.num_skipped += 1
        if reason is None:
            reason = f"Output: {output_file.name} already exists and shall not be overwritten\n"
        self.text.append(f"[SKIPPED] {datetime.datetime.now()} - Input: {input_file.name}; \n"
                         + reason)
    
    
    def log(self, outcome: ConversionOutcome) -> None:
        if outcome.successful:
            self.num_successful += 1
            self.text.append(f"[SUCCESS] {datetime.datetime.now()} - Input: {outcome.input_file.name}; Output: {outcome.output_file.name}:\n") 
            

            if outcome.warning_messages:
                self.num_warned_successful += 1
                for warning_message in outcome.warning_messages:
                    self.text.append(f"       [WARNING] {warning_message}\n")

        else:
            self.num_errors += 1
            if outcome.go_on:
                self.text.append(f"[ERROR] {datetime.datetime.now()} - Input: {outcome.input_file.name}\n:" + 
                                 f"         Error: {outcome.error_message}\n")
            else:
                self.has_halted = True
                self.text.append(f"[HALT] {datetime.datetime.now()} - ON {outcome.input_file.name}\n"+ 
                                        f"         Error: {outcome.error_message}\n")
                self.commit()
                raise Log.HaltError(f"Critical failure since conversion of {outcome.input_file.name}. All upcoming conversions aborted.\n" + outcome.error_message)
    
    @property
    def stats(self):
        return {
        "num_attempted": self.num_attempted,
        "share_successful": self.num_successful / self.num_attempted if self.num_attempted else 0,
        "share_skipped": self.num_skipped / self.num_attempted if self.num_attempted else 0,
        "share_errors": self.num_errors / self.num_attempted if self.num_attempted else 0,
        "share_warned_from_successful": self.num_warned_successful / self.num_successful if self.num_successful else 0,
        "has_halted": self.has_halted
        }
    
    def evaluation(self):
        self.text[1:1] = [
            f"{key}: {value * 100:.2f}%\n" if isinstance(value, float) else f"{key}: {value}\n"
            for key, value in self.stats.items()
        ]
        index = len(self.stats) + 1
        self.text[index:index] = "\n"
    
    def commit(self):
        self.evaluation()
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("".join(self.text))