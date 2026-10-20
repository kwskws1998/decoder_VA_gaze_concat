import sys
import pathlib

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
import math
import warnings
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
import torch
import gc
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore")
from models.reward_model_factory import ModelFactory
from trainers.reward_trainer import (
    RewardTrainerConstructor,
)
import functools
import wandb
import os
from trl import RewardTrainer, RewardConfig
from peft import PeftModel
from transformers import TrainerCallback


def wandb_hp_space(trial):
    return {
        "method": "bayes",
        "metric": {"name": "objective", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [5e-6]},
            "per_device_train_batch_size": {"values": [8]},
            "lr_scheduler_type": {"values": ["constant_with_warmup"]},
            "weight_decay": {"values": [0.001]},
            "adam_beta1": {"values": [0.9]},
            "adam_beta2": {"values": [0.999]},
        },
    }


def cosine_scheduler_with_min_lr_ratio(
    optimizer, num_training_steps, warmup_ratio=0.1, min_lr_ratio=0.8
):
    print(f"min_lr_ratio: {min_lr_ratio}")
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)


def model_init_func(
    model_name,
    bnb_config,
    input_layer,
    freeze_layer,
    freeze,
    use_softprompt,
    concat,
    noise_factor=0.0,
    peft_config=None,
    load_local_folder_name=None,
    fp_dropout=[0.0, 0.3],
    fixations_model_version=1,
    load_fix_model=True,
    features_used=[1, 1, 1, 1, 1],
    init_sigma_left=1.0,
    init_sigma_right=1.0,
    use_asym_gaussian_redistributor=True,
):
    global model
    try:
        del model
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    factory = ModelFactory(
        model_name=model_name,
        bnb_config=bnb_config,
        input_layer=input_layer,
        freeze_layer=freeze_layer,
        freeze=freeze,
        use_softprompt=use_softprompt,
        concat=concat,
        noise_factor=noise_factor,
        load_local_folder_name=load_local_folder_name,
        fp_dropout=fp_dropout,
        fixations_model_version=fixations_model_version,
        load_fix_model=load_fix_model,
        features_used=features_used,
        init_sigma_left=init_sigma_left,
        init_sigma_right=init_sigma_right,
        use_asym_gaussian_redistributor=use_asym_gaussian_redistributor,
    )

    model = factory.create_model()

    if load_local_folder_name is None:
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
            # model.modules_to_save.update([ model.base_model.fixations_embedding_projector,model.base_model.norm_layer_fix])

    else:
        model = PeftModel.from_pretrained(model, load_local_folder_name)
        if concat is False or use_softprompt is True:
            if load_local_folder_name[-1] == "/":
                load_local_folder_name = load_local_folder_name[:-1]
            model.base_model.model.fixations_embedding_projector.load_state_dict(
                torch.load(
                    load_local_folder_name + "/fixations_projector_state_dict.bin"
                )
            )
            model.base_model.model.norm_layer_fix.load_state_dict(
                torch.load(load_local_folder_name + "/layer_norm_state_dict.bin")
            )

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        if hasattr(model.base_model.model, "asym_gaussian_redistributor"):
            print("=" * 25)
            print("asym_gaussian_redistributor found in base_model.model")

    elif hasattr(model, "asym_gaussian_redistributor"):
        print("asym_gaussian_redistributor found in model")
        
    return model


class SigmaLoggingCallback(TrainerCallback):
    def __init__(self, redistributor):
        self.redistributor = redistributor

    def on_log(self, args, state, control, logs=None, **kwargs):
        redistributor = self.redistributor
        if hasattr(redistributor, "modules_to_save"):
            redistributor = redistributor.modules_to_save[redistributor._active_adapter]
        sigma_left = redistributor.sigma_left.item()
        sigma_right = redistributor.sigma_right.item()
        wandb.log(
            {
                "sigma/sigma_left": sigma_left,
                "sigma/sigma_right": sigma_right,
                "sigma/gap": sigma_right - sigma_left,
            },
            commit=False,
        )


class CustomRewardTrainer(RewardTrainer):
    def save_model(self, output_dir=None, _internal_call=True):
        # Call the original save_model method to save the model and tokenizer
        super().save_model(output_dir, _internal_call)
        # check if output_dir ends with '/'
        if output_dir[-1] == "/":
            output_dir = output_dir[:-1]
        torch.save(
            self.model.base_model.model.fixations_embedding_projector.state_dict(),
            output_dir + "/fixations_projector_state_dict.bin",
        )
        torch.save(
            self.model.base_model.model.norm_layer_fix.state_dict(),
            output_dir + "/layer_norm_state_dic.bin",
        )

        print(f"Additional components saved to {output_dir}")


class RewardTrainerConstructorGeneral(RewardTrainerConstructor):
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_name="OpenAssistant/oasst1",
        dataset_split="",
        use_lora=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=False,
        concat=True,
        noise_factor=0.0,
        batch_size=64,
        train_epochs=2,
        grid_search=False,
        gradient_acum_steps=1,
        logging_steps=1,
        gradient_checkpointing=False,
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        min_lr_ratio=0.8,
        weight_decay=0.1,
        input_layer=[4],
        freeze_layer=None,
        freeze=False,
        use_softprompt=True,
        fold=0,
        subsample_percentage=1,
        seed=42,
        fp_dropout=[0.0, 0.3],
        fixations_model_version=1,
        load_fix_model=True,
        features_used=[1, 1, 1, 1, 1],
        max_length=10000,
        max_tokens=None,
        init_sigma_left=1.0,
        init_sigma_right=1.0,
        sigma_lr=5e-3,
        sigma_lr_scheduler_type="cosine",
        sigma_learnable=True,
        use_asym_gaussian_redistributor=True,
    ):
        super().__init__(
            model_name=model_name,
            use_lora=use_lora,
            use_quantization=use_quantization,
            dataset_name=dataset_name,
            batch_size=batch_size,
            train_epochs=train_epochs,
            gradient_acum_steps=gradient_acum_steps,
            logging_steps=logging_steps,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            min_lr_ratio=min_lr_ratio,
            weight_decay=weight_decay,
            dataset_split=dataset_split,
            fold=fold,
            subsample_percentage=subsample_percentage,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.lr_scheduler_types_implemented = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ]
        self.model_name_log += "datav6"
        self.input_layer = input_layer
        if not isinstance(self.input_layer, list):
            self.input_layer = list(self.input_layer)
        self.freeze = freeze
        self.freeze_layer = freeze_layer
        if self.freeze_layer is None and self.freeze is True:
            self.freeze_layer = max(self.input_layer)
        elif self.freeze is False:
            self.freeze_layer = None
        if isinstance(self.freeze_layer, list):
            self.freeze_layer = max(self.freeze_layer)
        self.concat = concat
        if self.concat:
            self.input_layer = None
        self.noise_factor = noise_factor
        self.use_softprompt = use_softprompt
        self.grid_search = grid_search
        self.seed = seed
        self.fp_dropout = fp_dropout
        self.fixations_model_version = fixations_model_version
        self.load_fix_model = load_fix_model
        self.features_used = features_used
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.init_sigma_left = init_sigma_left
        self.init_sigma_right = init_sigma_right
        self.sigma_lr = sigma_lr
        self.sigma_lr_scheduler_type = sigma_lr_scheduler_type
        self.sigma_learnable = sigma_learnable
        self.use_asym_gaussian_redistributor = use_asym_gaussian_redistributor
        print("=" * 25)
        print(f"RewardTrainerConstructorGeneral initialized")
        print(f"Configured sigma_lr = {sigma_lr}, sigma_learnable = {sigma_learnable}")
        print("=" * 25)

    def train_model(self, train_samples=0, fold=None, save_folder="./reward_model", callbacks=None):
        if self.use_quantization:
            print("Config quantization")
            self.bnb_config = self.config_quantization()
        else:
            self.bnb_config = None

        if self.use_lora is True:
            print("Using LORA for trainer")
            self.config_lora(lora_r=8, lora_alpha=32, lora_dropout=0.1)
        else:
            self.peft_config = None

        model = model_init_func(
            model_name=self.model_name,
            bnb_config=self.bnb_config,
            input_layer=self.input_layer,
            freeze_layer=self.freeze_layer,
            freeze=self.freeze,
            use_softprompt=self.use_softprompt,
            concat=self.concat,
            noise_factor=self.noise_factor,
            peft_config=self.peft_config,
            fp_dropout=self.fp_dropout,
            fixations_model_version=self.fixations_model_version,
            load_fix_model=self.load_fix_model,
            features_used=self.features_used,
            init_sigma_left=self.init_sigma_left,
            init_sigma_right=self.init_sigma_right,
            use_asym_gaussian_redistributor=self.use_asym_gaussian_redistributor,
        )
        self.tokenizer = model.tokenizer
        self.model = model
        self.load_dataset(
            train_samples=train_samples,
            split=self.dataset_split,
            eval_mode=False,
            max_length=self.max_length,
            max_tokens=self.max_tokens,
        )

        self.set_trainer(save_folder=save_folder, callbacks=callbacks)
        if self.grid_search:
            self.best_trial = self.trainer.hyperparameter_search(
                direction="maximize",
                backend="wandb",
                hp_space=wandb_hp_space,
                # n_trials=1,
            )
        else:
            self.trainer.train()

    def eval_model(self, folder_name, mode="all"):
        if self.use_quantization:
            print("Config quantization")
            self.bnb_config = self.config_quantization()
        else:
            self.bnb_config = None

        self.model = model_init_func(
            model_name=self.model_name,
            bnb_config=self.bnb_config,
            input_layer=self.input_layer,
            freeze_layer=self.freeze_layer,
            freeze=self.freeze,
            use_softprompt=self.use_softprompt,
            concat=self.concat,
            fp_dropout=self.fp_dropout,
            fixations_model_version=self.fixations_model_version,
            load_fix_model=self.load_fix_model,
            features_used=self.features_used,
            load_local_folder_name=folder_name,
            use_asym_gaussian_redistributor=self.use_asym_gaussian_redistributor,
        )
        self.tokenizer = model.tokenizer
        if self.dataset_name == "allenai/reward-bench":
            self.load_dataset_rewardbench(
                max_length=self.max_length, max_tokens=self.max_tokens, mode=mode
            )
            if mode == "all":
                results = {}
                rb_all_data = self.test_dataset
                for subset_name, subset_data in rb_all_data.items():
                    self.test_dataset = subset_data
                    self.set_trainer_eval()
                    results[subset_name] = self.trainer.evaluate()
            else:
                self.test_dataset = rb_all_data[mode]
                self.set_trainer_eval()
                results = self.trainer.evaluate()
        else:
            self.load_dataset(
                split=self.dataset_split, eval_mode=True, max_length=self.max_length
            )
            self.set_trainer_eval()
            results = self.trainer.evaluate()
        return results

    def eval_model_v2(self):
        self.set_trainer_eval()
        results = self.trainer.evaluate()

        model = self.model
        base = getattr(model, "base_model", model)
        inner = getattr(base, "model", base)
        redistributor = getattr(inner, "asym_gaussian_redistributor", None)
        if redistributor is not None:
            if hasattr(redistributor, "modules_to_save"):
                redistributor = redistributor.modules_to_save[redistributor._active_adapter]
            sigma_left = redistributor.sigma_left.item()
            sigma_right = redistributor.sigma_right.item()
            results["eval_sigma_left"] = sigma_left
            results["eval_sigma_right"] = sigma_right
        else:
            sigma_left = sigma_right = None

        wandb.log({
            "final_eval/loss": results.get("eval_loss"),
            "final_eval/accuracy": results.get("eval_accuracy"),
            "final_eval/sigma_left": sigma_left,
            "final_eval/sigma_right": sigma_right,
        })

        return results

    def config_lora(self, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=["asym_gaussian_redistributor"],
        )
        return self.peft_config

    def set_training_args(self, save_folder="./reward_model"):
        training_args = {
            "output_dir": save_folder,
            "save_strategy": "steps",
            "save_steps": self.logging_steps,
            "metric_for_best_model": "accuracy",
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "logging_dir": "./logs",
            "report_to": "wandb",
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_acum_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "evaluation_strategy": "steps",
            "logging_steps": self.logging_steps,
            "num_train_epochs": self.train_epochs,
            "run_name": self.model_name_log,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": 0.02,
            "seed": self.seed,
        }

        if self.lr_scheduler_type in self.lr_scheduler_types_implemented:
            training_args["lr_scheduler_type"] = self.lr_scheduler_type
        training_args = RewardConfig(**training_args)
        return training_args

    def set_trainer_eval(self):
        training_args = self.set_training_args()
        training_args.report_to = []

        trainer_args = {
            "model": self.model,
            "args": training_args,
            "tokenizer": self.tokenizer,
            "eval_dataset": self.test_dataset,
        }

        # Initialize the RewardTrainer. if you are training the projecto create a new trainer that will save it
        if self.concat is False or self.use_softprompt is True:
            self.trainer = RewardTrainer(**trainer_args)
        else:
            self.trainer = RewardTrainer(**trainer_args)

    def set_trainer(self, save_folder="./reward_model", callbacks=None):
        wandb.login()
        os.environ["WANDB_PROJECT"] = self.wandb_project
        self.set_name_run()
        print(f"model_name_log: {self.model_name_log}")
        print(f"gradient_acum_steps: {self.gradient_acum_steps}")
        training_args = self.set_training_args(save_folder=save_folder)

        if self.grid_search:
            model_init = functools.partial(
                model_init_func,
                self.model_name,
                self.bnb_config,
                self.input_layer,
                self.freeze_layer,
                self.freeze,
                self.use_softprompt,
                self.concat,
                self.peft_config,
            )
        else:
            model_init = None

        trainer_args = {
            "model": self.model,
            "model_init": model_init,
            "args": training_args,
            "tokenizer": self.tokenizer,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "peft_config": self.peft_config,
        }

        sigma_callbacks = []
        model = self.model
        base = getattr(model, "base_model", model)
        inner = getattr(base, "model", base)
        redistributor = getattr(inner, "asym_gaussian_redistributor", None)
        if redistributor is not None:
            sigma_callbacks.append(SigmaLoggingCallback(redistributor))

        if callbacks is not None:
            trainer_args["callbacks"] = callbacks + sigma_callbacks
        elif sigma_callbacks:
            trainer_args["callbacks"] = sigma_callbacks

        self.load_optmizer_scheduler(len(self.train_dataset))
        trainer_args["optimizers"] = (self.optimizer, self.scheduler)

        if self.concat is False or self.use_softprompt is True:
            self.trainer = RewardTrainer(**trainer_args)
        else:
            self.trainer = RewardTrainer(**trainer_args)

    def load_optmizer_scheduler(self, num_samples):
        sigma_suffixes = ("sigma_left", "sigma_right")

        sigma_params = []
        base_params = []

        print("=" * 25, "\nSetting parameter configurations in optimizer\n")
        for name, p in self.model.named_parameters():
            if name.endswith(sigma_suffixes):
                if not self.sigma_learnable:
                    p.requires_grad_(False)
                    print("SIGMA PARAM FROZEN:", name)
                    continue
                if p.requires_grad:
                    sigma_params.append(p)
                    print("SIGMA PARAM DETECTED:", name)
            elif p.requires_grad:
                base_params.append(p)

        print("\nConfiguration done\n" + "=" * 25)

        param_groups = [
            {
                "params": base_params,
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
        ]
        if sigma_params:
            param_groups.append(
                {
                    "params": sigma_params,
                    "lr": self.sigma_lr,
                    "weight_decay": 0.0,
                }
            )

        self.optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

        num_training_steps = num_samples // self.batch_size * self.train_epochs

        num_warmup_steps = int(num_training_steps * 0.01)

        def base_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (1 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio

        if self.sigma_lr_scheduler_type == "constant":
            sigma_lr_lambda = lambda step: 1.0
        elif self.sigma_lr_scheduler_type == "linear":
            sigma_lr_lambda = lambda step: max(
                0.0, 1.0 - step / float(max(1, num_training_steps))
            )
        else:  # cosine (default)
            sigma_lr_lambda = base_lr_lambda

        lr_lambdas = [base_lr_lambda]
        if sigma_params:
            lr_lambdas.append(sigma_lr_lambda)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambdas)
