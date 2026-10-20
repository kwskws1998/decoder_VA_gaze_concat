import sys
import pathlib

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)

from models.reward_model_general_sp import (
    create_dynamic_class_RewardConcatenate,
)
from models.reward_model_general_add import (
    create_dynamic_class_RewardAdd,
)

from transformers import (
    LlamaForSequenceClassification,
    MistralForSequenceClassification,
)


class ModelFactory:
    def __init__(
        self,
        model_name,
        bnb_config=None,
        input_layer=None,
        freeze_layer=None,
        freeze=None,
        use_softprompt=None,
        concat=False,
        noise_factor=0.0,
        load_local_folder_name=None,
        fp_dropout=[0.0, 0.3],
        fixations_model_version=1,
        load_fix_model=True,
        features_used=[1, 1, 1, 1, 1],
        init_sigma_left=1.0,
        init_sigma_right=1.0,
        use_asym_gaussian_redistributor=True,
    ):
        self.model_name = model_name
        self.bnb_config = bnb_config
        self.input_layer = input_layer
        self.freeze_layer = freeze_layer
        self.noise_factor = noise_factor
        self.freeze = freeze
        self.use_softprompt = use_softprompt
        self.concat = concat
        self.load_local_folder_name = load_local_folder_name
        self.fp_dropout = fp_dropout
        self.fixations_model_version = fixations_model_version
        self.load_fix_model = load_fix_model
        self.features_used = features_used
        self.init_sigma_left = init_sigma_left
        self.init_sigma_right = init_sigma_right
        self.use_asym_gaussian_redistributor = use_asym_gaussian_redistributor

    def create_model(self):
        base_class = LlamaForSequenceClassification

        if self.concat:
            MyDynamicClass = create_dynamic_class_RewardConcatenate(base_class)
            return MyDynamicClass(
                model_name=self.model_name,
                bnb_config=self.bnb_config,
                use_softprompt=self.use_softprompt,
                load_local_folder_name=self.load_local_folder_name,
                noise_factor=self.noise_factor,
                fp_dropout=self.fp_dropout,
                fixations_model_version=self.fixations_model_version,
                load_fix_model=self.load_fix_model,
                features_used=self.features_used,
                init_sigma_left=self.init_sigma_left,
                init_sigma_right=self.init_sigma_right,
                use_asym_gaussian_redistributor=self.use_asym_gaussian_redistributor,
            )
        MyDynamicClass = create_dynamic_class_RewardAdd(base_class)
        return MyDynamicClass(
            model_name=self.model_name,
            bnb_config=self.bnb_config,
            load_local_folder_name=self.load_local_folder_name,
            noise_factor=self.noise_factor,
            fp_dropout=self.fp_dropout,
            fixations_model_version=self.fixations_model_version,
            features_used=self.features_used,
            init_sigma_left=self.init_sigma_left,
            init_sigma_right=self.init_sigma_right,
            use_asym_gaussian_redistributor=self.use_asym_gaussian_redistributor,
        )
