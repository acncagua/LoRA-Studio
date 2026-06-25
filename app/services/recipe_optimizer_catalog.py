from __future__ import annotations

import json
from typing import Any


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


OPTIMIZER_DEFINITIONS_V2 = [
    {
        "id": "AdamW8bit",
        "name": "AdamW8bit",
        "display_name": "AdamW8bit",
        "category": "stable",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.0001,
        "default_unet_lr": 0.0001,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine", "linear", "constant_with_warmup"],
        "optimizer_args_schema": {},
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_steps_confidence": "medium",
        "description": "安定した標準Optimizer。最初の比較基準。",
        "risk_note": "安定した標準Optimizer。最初の比較基準。",
        "compatibility_notes": ["通常LR指定で扱います。"],
    },
    {
        "id": "PagedAdamW8bit",
        "name": "PagedAdamW8bit",
        "display_name": "PagedAdamW8bit",
        "category": "stable/memory_saving",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.0001,
        "default_unet_lr": 0.0001,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine", "linear", "constant_with_warmup"],
        "optimizer_args_schema": {},
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_steps_confidence": "medium",
        "description": "AdamW8bitに近い省メモリ候補。",
        "risk_note": "bitsandbytes環境依存に注意してください。",
        "compatibility_notes": ["AdamW8bitに近い目安です。"],
    },
    {
        "id": "Adafactor",
        "name": "Adafactor",
        "display_name": "Adafactor",
        "category": "memory_saving/advanced",
        "lr_semantics": "relative_step",
        "default_learning_rate": None,
        "default_unet_lr": None,
        "default_text_encoder_lr": 0,
        "default_scheduler": "adafactor",
        "allowed_schedulers": ["adafactor", "constant_with_warmup", "constant"],
        "optimizer_args_schema": {"relative_step": "bool", "scale_parameter": "bool", "warmup_init": "bool"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "省メモリ寄りのAdvanced optimizer。",
        "risk_note": "relative_step=Trueでは通常LRとは意味が異なります。",
        "compatibility_notes": ["Auto profileとFixed profileでLR意味が変わります。"],
    },
    {
        "id": "Lion",
        "name": "Lion",
        "display_name": "Lion",
        "category": "experimental",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.00005,
        "default_unet_lr": 0.00005,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "実験的Optimizer。AdamWに似たUIでも挙動は異なります。",
        "risk_note": "効きが強めに出ることがあります。",
        "compatibility_notes": ["balancedは5e-5を初期値にします。"],
    },
    {
        "id": "Lion8bit",
        "name": "Lion8bit",
        "display_name": "Lion8bit",
        "category": "experimental/memory_saving",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.00005,
        "default_unet_lr": 0.00005,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "Lionの8bit候補。",
        "risk_note": "bitsandbytes等の依存状況に注意。",
        "compatibility_notes": ["availabilityは環境検出に依存します。"],
    },
    {
        "id": "PagedLion8bit",
        "name": "PagedLion8bit",
        "display_name": "PagedLion8bit",
        "category": "experimental/memory_saving",
        "lr_semantics": "normal_lr",
        "default_learning_rate": 0.00005,
        "default_unet_lr": 0.00005,
        "default_text_encoder_lr": 0,
        "default_scheduler": "cosine",
        "allowed_schedulers": ["constant", "cosine"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "Paged Lionの8bit候補。",
        "risk_note": "bitsandbytes等の依存状況に注意。",
        "compatibility_notes": ["availabilityは環境検出に依存します。"],
    },
    {
        "id": "DAdaptAdam",
        "name": "DAdaptAdam",
        "display_name": "DAdaptAdam",
        "category": "auto_lr/advanced",
        "lr_semantics": "auto_lr_multiplier",
        "default_learning_rate": 1.0,
        "default_unet_lr": 1.0,
        "default_text_encoder_lr": 0,
        "default_scheduler": "constant",
        "allowed_schedulers": ["constant"],
        "optimizer_args_schema": {"decouple": "bool", "weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "自動LR系のAdvanced optimizer。",
        "risk_note": "learning_rate=1.0は通常LRではなく自動推定LRへの倍率です。",
        "compatibility_notes": ["schedulerはconstantを推奨します。"],
    },
    {
        "id": "DAdaptLion",
        "name": "DAdaptLion",
        "display_name": "DAdaptLion",
        "category": "auto_lr/experimental",
        "lr_semantics": "auto_lr_multiplier",
        "default_learning_rate": 1.0,
        "default_unet_lr": 1.0,
        "default_text_encoder_lr": 0,
        "default_scheduler": "constant",
        "allowed_schedulers": ["constant"],
        "optimizer_args_schema": {"weight_decay": "float"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "DAdapt + Lion系の実験的Optimizer。",
        "risk_note": "比較用。挙動差と固定化に注意してください。",
        "compatibility_notes": ["schedulerはconstantを推奨します。"],
    },
    {
        "id": "Prodigy",
        "name": "Prodigy",
        "display_name": "Prodigy",
        "category": "auto_lr/advanced",
        "lr_semantics": "auto_lr_multiplier",
        "default_learning_rate": 1.0,
        "default_unet_lr": 1.0,
        "default_text_encoder_lr": 0,
        "default_scheduler": "constant",
        "allowed_schedulers": ["constant"],
        "optimizer_args_schema": {"decouple": "bool", "weight_decay": "float", "d_coef": "float", "use_bias_correction": "bool"},
        "target_steps_min": 2500,
        "target_steps_recommended": 4000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "自動LR系。短期収束の比較候補。",
        "risk_note": "強く効くことがあるため過学習注意。",
        "compatibility_notes": ["schedulerはconstantを推奨します。"],
    },
    {
        "id": "Custom",
        "name": "Custom",
        "display_name": "Custom",
        "category": "custom",
        "lr_semantics": "custom",
        "default_learning_rate": None,
        "default_unet_lr": None,
        "default_text_encoder_lr": None,
        "default_scheduler": "custom",
        "allowed_schedulers": [],
        "optimizer_args_schema": {},
        "target_steps_min": 2500,
        "target_steps_recommended": 5000,
        "target_steps_max": 8000,
        "target_steps_confidence": "low",
        "description": "手動指定用。",
        "risk_note": "互換性チェック対象外の設定を含む可能性があります。",
        "compatibility_notes": ["Raw Args利用時はsd-scripts仕様を確認してください。"],
    },
]


OPTIMIZER_PROFILES_V2 = [
    ("adamw8bit_sdxl_balanced", "AdamW8bit", "SDXL", "balanced", "AdamW8bit Balanced", 0.0001, 0.0001, 0, "constant", {}, 2500, 5000, 8000, "SDXL標準の比較基準。", "安定した標準Optimizer。最初の比較基準。"),
    ("paged_adamw8bit_sdxl_memory", "PagedAdamW8bit", "SDXL", "memory_saving", "PagedAdamW8bit SDXL Memory", 0.0001, 0.0001, 0, "cosine", {}, 2500, 5000, 8000, "省メモリ寄りのAdamW候補。", "環境依存に注意。"),
    ("paged_adamw8bit_sdxl_balanced", "PagedAdamW8bit", "SDXL", "memory_saving", "PagedAdamW8bit Balanced", 0.0001, 0.0001, 0, "constant", {}, 2500, 5000, 8000, "AdamW8bitに近い省メモリ候補。", "まずSmokeで環境対応を確認。"),
    ("adafactor_sdxl_auto", "Adafactor", "SDXL", "auto_lr", "Adafactor Auto", None, None, 0, "adafactor", {"relative_step": True, "scale_parameter": True, "warmup_init": True}, 2500, 4000, 8000, "Adafactor relative_step運用。", "通常LRとは意味が異なります。"),
    ("adafactor_sdxl_fixed", "Adafactor", "SDXL", "advanced", "Adafactor Fixed", 0.0001, 0.0001, 0, "constant_with_warmup", {"relative_step": False, "scale_parameter": False, "warmup_init": False}, 2500, 5000, 8000, "Adafactor固定LR運用。", "max_grad_norm=0.0推奨。AdamWとは挙動が異なります。"),
    ("lion_sdxl_soft", "Lion", "SDXL", "experimental", "Lion Soft", 0.00005, 0.00005, 0, "constant", {"weight_decay": 0.01}, 2500, 4000, 8000, "Lionを弱めに試す実験profile。", "実験的Optimizer。最初はLR低め。"),
    ("lion_sdxl_balanced", "Lion", "SDXL", "balanced", "Lion SDXL Balanced", 0.00005, 0.00005, 0, "cosine", {"weight_decay": 0.01}, 2500, 4000, 8000, "Lionを弱めに試すbalanced profile。", "実験的です。"),
    ("lion_sdxl_balanced_experimental", "Lion", "SDXL", "experimental", "Lion Balanced Experimental", 0.0001, 0.0001, 0, "constant", {"weight_decay": 0.01}, 2500, 4000, 8000, "Lionの標準寄り実験profile。", "Softで弱い場合に比較。"),
    ("dadapt_adam_sdxl_auto", "DAdaptAdam", "SDXL", "auto_lr", "DAdaptAdam SDXL Auto", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01}, 2500, 4000, 8000, "DAdaptAdam自動LR倍率運用。", "learning_rate=1.0は倍率です。"),
    ("dadaptadam_sdxl_auto", "DAdaptAdam", "SDXL", "auto_lr", "DAdaptAdam Auto", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01}, 2500, 4000, 8000, "DAdaptAdam自動LR倍率運用。AdamW-style decoupled decay。", "learning_rate=1.0は通常LRではなく倍率です。"),
    ("dadapt_lion_sdxl_auto", "DAdaptLion", "SDXL", "experimental", "DAdaptLion SDXL Auto", 1.0, 1.0, 0, "constant", {"weight_decay": 0.01}, 2500, 4000, 8000, "DAdaptLion実験profile。", "実験的です。"),
    ("dadaptlion_sdxl_auto", "DAdaptLion", "SDXL", "experimental", "DAdaptLion Auto", 1.0, 1.0, 0, "constant", {"weight_decay": 0.01}, 2500, 4000, 8000, "DAdapt + Lion系の実験profile。", "DAdaptAdamより検証優先度は低め。"),
    ("prodigy_sdxl_auto", "Prodigy", "SDXL", "auto_lr", "Prodigy SDXL Auto", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01, "d_coef": 1.0, "use_bias_correction": True}, 2500, 4000, 8000, "Prodigy自動LR運用。", "強く効くことがあります。"),
    ("prodigy_sdxl_soft", "Prodigy", "SDXL", "advanced", "Prodigy SDXL Soft", 1.0, 1.0, 0, "constant", {"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}, 2500, 4000, 8000, "Prodigyを弱めに試すprofile。", "固定化に注意してください。"),
    ("adamw8bit_sd15_balanced", "AdamW8bit", "SD15", "balanced", "AdamW8bit SD1.5 Balanced", 0.0001, 0.0001, 0, "cosine", {}, 2000, 4000, 7000, "SD1.5標準の比較基準。", "SDXLより軽めのstep目安です。"),
]


NETWORK_TYPES = [
    ("standard_lora", "standard_lora", "Standard LoRA", "networks.lora", "standard_lora", "available", {"network_dim": "int", "network_alpha": "int"}, 32, 16, "標準LoRA。安定した通常実行対象です。", "安定した標準network type。"),
    (
        "lora_c3lier",
        "lora_c3lier",
        "LoRA-C3Lier",
        "networks.lora",
        "lora_c3lier",
        "available",
        {
            "network_dim": "int",
            "network_alpha": "int",
            "conv_dim": "int",
            "conv_alpha": "int",
            "alias": ["LoCon-like", "LoCon相当"],
            "display_name_ja": "LoRA-C3Lier（セリア）",
            "future_network_types": ["lycoris_locon"],
        },
        32,
        16,
        "sd-scripts標準LoRAを3x3 Conv2dにも拡張する方式。networks.loraにconv_dim / conv_alphaを指定します。LyCORIS LoConとは別実装として扱います。",
        "Phase 12.5の最初のNetwork Type対応対象。LoCon-likeですが、LyCORIS LoConではありません。Recipe整備は段階追加します。",
    ),
    ("loha", "loha", "LoHa", "loha", "loha", "planned", {}, 32, 16, "後続Phaseで対応予定。Phase 12.5では対象外です。", "Phase 12.5では実行不可。"),
    ("lokr", "lokr", "LoKr", "lokr", "lokr", "planned", {}, 32, 16, "後続Phaseで対応予定。Phase 12.5では対象外です。", "Phase 12.5では実行不可。"),
    ("lycoris_locon", "lycoris_locon", "LyCORIS LoCon", "lycoris.kohya", "lycoris_locon", "planned", {"algo": "locon"}, 32, 16, "後続Phaseで対応予定。lycoris.kohya + algo=locon の別実装として扱います。", "Phase 12.5では対象外です。"),
    ("lycoris", "lycoris", "LyCORIS", "lycoris.kohya", "lycoris", "planned", {}, 32, 16, "後続Phaseで対応予定。LoHa / LoKr / IA3 / DyLoRAなどはPhase 12.5対象外です。", "Phase 12.5では実行不可。"),
    ("custom", "custom", "Custom", "", "custom", "unsupported", {}, None, None, "手動定義用。", "互換性チェック対象外です。"),
]


TRAINING_PURPOSES = [
    ("character_face", "character_face", "Character Face", "顔LoRA向け。", ["face_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 10),
    ("character_full_body", "character_full_body", "Character Full Body", "全身・衣装を含むキャラクター向け。", ["full_body_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 20),
    ("style", "style", "Style", "画風LoRA向け。", ["style_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 30),
    ("costume", "costume", "Costume", "衣装LoRA向け。", ["costume_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 40),
    ("object", "object", "Object", "物体LoRA向け。", ["object_reference", "dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 50),
    ("concept", "concept", "Concept", "概念LoRA向け。", ["dataset_sample"], "standard_validation_v1", "sdxl_face_basic_3prompts", {}, 60),
    ("custom", "custom", "Custom", "完全カスタム用途。", [], None, None, {}, 999),
]


COMMON_SDXL_PARAMS = {
    "network_module": "networks.lora",
    "save_model_as": "safetensors",
    "save_every_n_epochs": 1,
    "sample_every_n_epochs": 1,
    "generate_training_samples": True,
    "sample_sampler": "euler_a",
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "cache_latents": True,
    "gradient_checkpointing": True,
    "max_data_loader_n_workers": 2,
    "persistent_data_loader_workers": True,
    "resolution": [1024, 1024],
    "clip_skip": 2,
}


def sdxl_params(**overrides: Any) -> dict[str, Any]:
    params = dict(COMMON_SDXL_PARAMS)
    params.update(overrides)
    return params


COMMON_SD15_PARAMS = {
    **COMMON_SDXL_PARAMS,
    "resolution": [512, 512],
    "clip_skip": 2,
}


def sd15_params(**overrides: Any) -> dict[str, Any]:
    params = dict(COMMON_SD15_PARAMS)
    params.update(overrides)
    return params


OPTIMIZER_DEFINITION_EXTRAS: dict[str, dict[str, Any]] = {
    "AdamW8bit": {
        "sd_scripts_optimizer_type": "AdamW8bit",
        "aliases": [],
        "required_dependencies": ["bitsandbytes"],
        "lr_semantics_help": "通常learning rateです。1e-4前後を基準にします。",
        "smoke_test_priority": 1,
    },
    "PagedAdamW8bit": {
        "sd_scripts_optimizer_type": "PagedAdamW8bit",
        "aliases": [],
        "required_dependencies": ["bitsandbytes"],
        "lr_semantics_help": "AdamW8bitに近い通常LRです。省メモリ候補として扱います。",
        "smoke_test_priority": 2,
    },
    "Prodigy": {
        "sd_scripts_optimizer_type": "Prodigy",
        "aliases": [],
        "required_dependencies": ["prodigyopt"],
        "lr_semantics_help": "learning_rate=1.0は通常LRではなくAuto-LR倍率です。",
        "smoke_test_priority": 3,
    },
    "Adafactor": {
        "sd_scripts_optimizer_type": "Adafactor",
        "aliases": ["AdaFactor", "adafactor"],
        "required_dependencies": ["transformers"],
        "lr_semantics_help": "relative_step=Trueではlearning_rateをAdafactorが自動調整します。",
        "smoke_test_priority": 4,
    },
    "Lion": {
        "sd_scripts_optimizer_type": "Lion",
        "aliases": [],
        "required_dependencies": ["lion-pytorch"],
        "lr_semantics_help": "通常LR型ですがExperimentalです。AdamWと挙動が異なります。",
        "smoke_test_priority": 6,
    },
    "DAdaptAdam": {
        "sd_scripts_optimizer_type": "DAdaptAdam",
        "aliases": [],
        "required_dependencies": ["dadaptation"],
        "lr_semantics_help": "learning_rate=1.0はDAdaptationが推定したLRへの倍率です。",
        "smoke_test_priority": 7,
    },
    "DAdaptLion": {
        "sd_scripts_optimizer_type": "DAdaptLion",
        "aliases": [],
        "required_dependencies": ["dadaptation"],
        "lr_semantics_help": "learning_rate=1.0はAuto-LR倍率です。DAdapt + LionのExperimental候補です。",
        "smoke_test_priority": 8,
    },
}


def optimizer_definition_extra(item: dict[str, Any]) -> dict[str, Any]:
    extra = OPTIMIZER_DEFINITION_EXTRAS.get(item["id"], {})
    required_schema = {
        "optimizer_type": extra.get("sd_scripts_optimizer_type") or item["id"],
        "lr_scheduler": item.get("default_scheduler"),
        "optimizer_args": item.get("optimizer_args_schema") or {},
    }
    recommended_schema = {
        "target_steps_min": item.get("target_steps_min"),
        "target_steps_recommended": item.get("target_steps_recommended"),
        "target_steps_max": item.get("target_steps_max"),
    }
    return {
        "sd_scripts_optimizer_type": extra.get("sd_scripts_optimizer_type") or item["id"],
        "aliases": extra.get("aliases", []),
        "required_dependencies": extra.get("required_dependencies", []),
        "required_params_schema": required_schema,
        "recommended_params_schema": recommended_schema,
        "lr_semantics_help": extra.get("lr_semantics_help") or item.get("description") or "",
        "smoke_test_priority": extra.get("smoke_test_priority", 999),
        "validated_optimizer_type": None,
        "validation_status": "untested",
    }


def optimizer_profile_param_bundle(row: tuple[Any, ...]) -> dict[str, Any]:
    (
        _profile_id,
        optimizer_definition_id,
        _model_family,
        _profile_type,
        _display_name,
        learning_rate,
        unet_lr,
        text_encoder_lr,
        scheduler,
        args,
        target_min,
        target_recommended,
        target_max,
        _description,
        _risk_note,
    ) = row
    definition_extra = OPTIMIZER_DEFINITION_EXTRAS.get(optimizer_definition_id, {})
    optimizer_type = definition_extra.get("sd_scripts_optimizer_type") or optimizer_definition_id
    required_params = {
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "unet_lr": unet_lr,
        "text_encoder_lr": text_encoder_lr,
        "lr_scheduler": scheduler,
        "optimizer_args": args,
    }
    command_params = {
        "optimizer_type": optimizer_type,
        "lr_scheduler": scheduler,
        "text_encoder_lr1": text_encoder_lr,
        "text_encoder_lr2": 0 if text_encoder_lr == 0 else text_encoder_lr,
        "optimizer_args": args,
    }
    if learning_rate is not None:
        command_params["learning_rate"] = learning_rate
    if unet_lr is not None:
        command_params["unet_lr"] = unet_lr
    if optimizer_definition_id == "Adafactor" and args.get("relative_step") is False:
        command_params["max_grad_norm"] = 0.0
    recommended_params = {
        "target_steps_min": target_min,
        "target_steps_recommended": target_recommended,
        "target_steps_max": target_max,
        "optimizer_args": args,
    }
    smoke_params = {
        **command_params,
        "max_train_steps": 2,
        "network_dim": 4,
        "network_alpha": 2,
        "save_every_n_steps": 1,
        "sample_every_n_steps": 1,
        "resolution": [512, 512],
    }
    return {
        "sd_scripts_optimizer_type": optimizer_type,
        "required_params": required_params,
        "recommended_params": recommended_params,
        "command_params": command_params,
        "smoke_params": smoke_params,
    }


OPTIMIZER_DESCRIPTION_EN: dict[str, str] = {
    "AdamW8bit": "Stable 8-bit AdamW optimizer. Use normal learning rates around 1e-4 as a baseline.",
    "PagedAdamW8bit": "Memory-saving AdamW8bit variant. Use normal learning rates and confirm environment support.",
    "Prodigy": "Auto-LR optimizer using learning_rate as a multiplier, commonly 1.0 rather than a normal LR.",
    "Adafactor": "Memory-oriented optimizer. relative_step profiles manage LR differently from normal LR profiles.",
    "Lion": "Experimental normal-LR optimizer with behavior different from AdamW. Use for comparison first.",
    "DAdaptAdam": "Auto-LR DAdaptation optimizer. learning_rate acts as a multiplier and dadaptation is required.",
    "DAdaptLion": "Experimental DAdaptation + Lion optimizer. Keep it as an advanced comparison profile.",
}


OPTIMIZER_RISK_NOTE_EN: dict[str, str] = {
    "AdamW8bit": "Recommended stable baseline. Watch overtraining when repeats or epochs are high.",
    "PagedAdamW8bit": "Useful for memory pressure, but verify support in the sd-scripts environment.",
    "Prodigy": "Auto-LR can become strong. Compare weights and watch for overtraining.",
    "Adafactor": "LR semantics differ by profile. Confirm whether relative_step or fixed LR is selected.",
    "Lion": "Experimental. Start with soft settings and compare against AdamW before relying on it.",
    "DAdaptAdam": "Requires dadaptation in the sd-scripts venv. Long-run behavior should be verified.",
    "DAdaptLion": "Experimental and dependency-sensitive. Use only after Smoke/Mini Pilot validation.",
}


def optimizer_description_en(item: dict[str, Any]) -> str:
    return OPTIMIZER_DESCRIPTION_EN.get(str(item.get("id") or ""), str(item.get("description") or ""))


def optimizer_risk_note_en(item: dict[str, Any]) -> str:
    return OPTIMIZER_RISK_NOTE_EN.get(str(item.get("id") or ""), str(item.get("risk_note") or ""))


def optimizer_profile_description_en(optimizer_id: str, profile_type: str) -> str:
    kind = str(profile_type or "").replace("_", " ")
    if optimizer_id == "Prodigy":
        return f"Prodigy {kind} profile. Uses auto-LR multiplier settings for advanced comparison."
    if optimizer_id == "Adafactor":
        return f"Adafactor {kind} profile. Check whether it uses relative_step or fixed LR."
    if optimizer_id == "Lion":
        return f"Lion {kind} experimental profile. Use as a comparison against AdamW."
    if optimizer_id == "DAdaptAdam":
        return f"DAdaptAdam {kind} profile. Requires dadaptation and benefits from longer validation."
    if optimizer_id == "DAdaptLion":
        return f"DAdaptLion {kind} experimental profile. Requires dadaptation and careful validation."
    if optimizer_id == "PagedAdamW8bit":
        return f"PagedAdamW8bit {kind} profile for memory-saving AdamW-style training."
    return f"{optimizer_id} {kind} profile for stable baseline training."


def optimizer_profile_risk_note_en(optimizer_id: str) -> str:
    return OPTIMIZER_RISK_NOTE_EN.get(optimizer_id, "Review Smoke/Mini Pilot status before using this profile.")


TRAINING_RECIPES_V2 = [
    {
        "id": "sdxl_character_face_adamw8bit_smoke",
        "name": "sdxl_character_face_adamw8bit_smoke",
        "display_name": "SDXL Character Face / AdamW8bit Smoke",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "smoke",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=4, network_alpha=2, train_batch_size=1, repeats=1, max_train_steps=2, save_every_n_steps=1, sample_every_n_steps=1, resolution=[512, 512]),
        "basic_params": {"network_dim": 4, "network_alpha": 2, "train_batch_size": 1, "repeats": 1, "max_train_steps": 2},
        "advanced_params": {},
        "raw_args": {},
        "target": (1, 2, 5, 1),
        "expected_behavior": "実行経路のsmoke確認用。",
        "risk_note": "品質評価には使いません。",
        "sort_order": 10,
    },
    {
        "id": "sdxl_character_face_adamw8bit_pilot_3epoch",
        "name": "sdxl_character_face_adamw8bit_pilot_3epoch",
        "display_name": "SDXL Character Face / AdamW8bit Pilot 3 Epoch",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "pilot",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=2, max_train_epochs=3, min_snr_gamma=5),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 2, "max_train_epochs": 3},
        "advanced_params": {"min_snr_gamma": 5},
        "raw_args": {},
        "target": (100, 150, 300, 3),
        "expected_behavior": "短時間でloss推移とepoch差を見る。",
        "risk_note": "完成品質ではありません。",
        "sort_order": 20,
    },
    {
        "id": "sdxl_character_face_adamw8bit_standard_6epoch",
        "name": "sdxl_character_face_adamw8bit_standard_6epoch",
        "display_name": "SDXL Character Face / AdamW8bit Standard 6 Epoch",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "balanced",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=6, min_snr_gamma=5, no_metadata=True),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 6},
        "advanced_params": {"min_snr_gamma": 5, "no_metadata": True},
        "raw_args": {},
        "target": (2500, 5000, 8000, 6),
        "expected_behavior": "本番寄り標準候補。",
        "risk_note": "候補epochを確認してください。",
        "sort_order": 30,
    },
    {
        "id": "sdxl_character_face_adamw8bit_standard_10epoch",
        "name": "sdxl_character_face_adamw8bit_standard_10epoch",
        "display_name": "SDXL Character Face / AdamW8bit Standard 10 Epoch",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "balanced_long",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=10),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 10},
        "advanced_params": {},
        "raw_args": {},
        "target": (2500, 5000, 8000, 10),
        "expected_behavior": "標準10epoch。候補比較前提。",
        "risk_note": "長めなので保存間隔に注意してください。",
        "sort_order": 40,
    },
    {
        "id": "sdxl_character_face_adamw8bit_generalize",
        "name": "sdxl_character_face_adamw8bit_generalize",
        "display_name": "SDXL Character Face / AdamW8bit Generalize",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "AdamW8bit",
        "optimizer_profile_id": "adamw8bit_sdxl_balanced",
        "network_type_id": "standard_lora",
        "recipe_type": "generalize",
        "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.00005, unet_lr=0.00005, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=16, network_alpha=8, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 16, "network_alpha": 8, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {},
        "raw_args": {},
        "target": (2500, 5000, 8000, 6),
        "expected_behavior": "弱めで固定化を避ける。",
        "risk_note": "顔再現が弱い場合があります。",
        "sort_order": 50,
    },
    {
        "id": "sdxl_character_face_lion_balanced_experimental",
        "name": "sdxl_character_face_lion_balanced_experimental",
        "display_name": "SDXL Character Face / Lion Balanced / Experimental",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Lion",
        "optimizer_profile_id": "lion_sdxl_balanced_experimental",
        "network_type_id": "standard_lora",
        "recipe_type": "experimental",
        "params": sdxl_params(optimizer_type="Lion", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"weight_decay": 0.01}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "Lionを弱めに比較する。",
        "risk_note": "Experimental。実学習評価は未完了。",
        "sort_order": 100,
    },
    {
        "id": "sdxl_character_face_adafactor_auto_advanced",
        "name": "sdxl_character_face_adafactor_auto_advanced",
        "display_name": "SDXL Character Face / Adafactor Auto / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Adafactor",
        "optimizer_profile_id": "adafactor_sdxl_auto",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="Adafactor", lr_scheduler="adafactor", text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"relative_step": True, "scale_parameter": True, "warmup_init": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"relative_step": True, "scale_parameter": True, "warmup_init": True}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "Adafactor Autoを比較する。",
        "risk_note": "Advanced。通常LRとは意味が異なります。",
        "sort_order": 110,
    },
    {
        "id": "sdxl_character_face_adafactor_fixed_advanced",
        "name": "sdxl_character_face_adafactor_fixed_advanced",
        "display_name": "SDXL Character Face / Adafactor Fixed / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Adafactor",
        "optimizer_profile_id": "adafactor_sdxl_fixed",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="Adafactor", lr_scheduler="constant_with_warmup", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, max_grad_norm=0.0, optimizer_args={"relative_step": False, "scale_parameter": False, "warmup_init": False}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"max_grad_norm": 0.0, "optimizer_args": {"relative_step": False, "scale_parameter": False, "warmup_init": False}},
        "raw_args": {},
        "target": (2500, 5000, 8000, 6),
        "expected_behavior": "Adafactorを固定LRで使うfallback候補。",
        "risk_note": "Advanced。Autoが不安定な場合の比較候補です。",
        "sort_order": 112,
    },
    {
        "id": "sdxl_character_face_dadapt_adam_auto_advanced",
        "name": "sdxl_character_face_dadapt_adam_auto_advanced",
        "display_name": "SDXL Character Face / DAdaptAdam Auto / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "DAdaptAdam",
        "optimizer_profile_id": "dadaptadam_sdxl_auto",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="DAdaptAdam", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"decouple": True, "weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"decouple": True, "weight_decay": 0.01}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "DAdaptAdamを比較する。",
        "risk_note": "Advanced。learning_rate=1.0は倍率です。",
        "sort_order": 120,
    },
    {
        "id": "sdxl_character_face_dadapt_lion_auto_experimental",
        "name": "sdxl_character_face_dadapt_lion_auto_experimental",
        "display_name": "SDXL Character Face / DAdaptLion Auto / Experimental",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "DAdaptLion",
        "optimizer_profile_id": "dadaptlion_sdxl_auto",
        "network_type_id": "standard_lora",
        "recipe_type": "experimental",
        "params": sdxl_params(optimizer_type="DAdaptLion", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"weight_decay": 0.01}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "DAdapt + Lion系を比較する実験候補。",
        "risk_note": "Experimental。DAdaptAdamより検証優先度は低めです。",
        "sort_order": 125,
    },
    {
        "id": "sdxl_character_face_prodigy_soft_advanced",
        "name": "sdxl_character_face_prodigy_soft_advanced",
        "display_name": "SDXL Character Face / Prodigy Soft / Advanced",
        "model_family": "SDXL",
        "training_purpose_id": "character_face",
        "optimizer_definition_id": "Prodigy",
        "optimizer_profile_id": "prodigy_sdxl_soft",
        "network_type_id": "standard_lora",
        "recipe_type": "advanced",
        "params": sdxl_params(optimizer_type="Prodigy", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
        "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
        "advanced_params": {"optimizer_args": {"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}},
        "raw_args": {},
        "target": (2500, 4000, 8000, 6),
        "expected_behavior": "Prodigy Softを比較する。",
        "risk_note": "Advanced。強く効くことがあります。",
        "sort_order": 130,
    },
]


TRAINING_RECIPES_V2.extend(
    [
        {
            "id": "sdxl_character_face_lora_c3lier_adamw8bit_balanced",
            "name": "sdxl_character_face_lora_c3lier_adamw8bit_balanced",
            "display_name": "SDXL Character Face / LoRA-C3Lier / AdamW8bit Balanced",
            "model_family": "SDXL",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "lora_c3lier",
            "recipe_type": "balanced",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, conv_dim=8, conv_alpha=4, network_args={"conv_dim": 8, "conv_alpha": 4}, train_batch_size=1, repeats=10, max_train_epochs=10),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "conv_dim": 8, "conv_alpha": 4, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 10},
            "advanced_params": {"network_args": {"conv_dim": 8, "conv_alpha": 4}},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "LoRA-C3Lierで3x3 Conv層も学習するCharacter Face向け標準候補。",
            "risk_note": "LyCORIS LoConとは別実装です。conv_dim / conv_alphaを小さめにして比較してください。",
            "sort_order": 60,
        },
        {
            "id": "sdxl_style_lora_c3lier_adamw8bit_soft",
            "name": "sdxl_style_lora_c3lier_adamw8bit_soft",
            "display_name": "SDXL Style / LoRA-C3Lier / AdamW8bit Soft",
            "model_family": "SDXL",
            "training_purpose_id": "style",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "lora_c3lier",
            "recipe_type": "soft",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, conv_dim=8, conv_alpha=4, network_args={"conv_dim": 8, "conv_alpha": 4}, train_batch_size=1, repeats=10, max_train_epochs=10),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "conv_dim": 8, "conv_alpha": 4, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 10},
            "advanced_params": {"network_args": {"conv_dim": 8, "conv_alpha": 4}},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "LoRA-C3LierをStyle向けに弱めに比較する候補。",
            "risk_note": "3x3 Conv拡張で画風が強く出る場合があります。Standard LoRAと比較してください。",
            "sort_order": 205,
        },
        {
            "id": "sdxl_costume_lora_c3lier_adamw8bit_balanced",
            "name": "sdxl_costume_lora_c3lier_adamw8bit_balanced",
            "display_name": "SDXL Costume / LoRA-C3Lier / AdamW8bit Balanced",
            "model_family": "SDXL",
            "training_purpose_id": "costume",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "lora_c3lier",
            "recipe_type": "balanced",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, conv_dim=8, conv_alpha=4, network_args={"conv_dim": 8, "conv_alpha": 4}, train_batch_size=1, repeats=10, max_train_epochs=10),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "conv_dim": 8, "conv_alpha": 4, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 10},
            "advanced_params": {"network_args": {"conv_dim": 8, "conv_alpha": 4}},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "LoRA-C3Lierで衣装や細部形状の入りを比較する候補。",
            "risk_note": "LyCORIS LoConではありません。衣装固定化が強い場合はStandard LoRAへ戻してください。",
            "sort_order": 305,
        },
        {
            "id": "sdxl_character_face_adamw8bit_soft",
            "name": "sdxl_character_face_adamw8bit_soft",
            "display_name": "SDXL Character Face / AdamW8bit Soft",
            "model_family": "SDXL",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "soft",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.00007, unet_lr=0.00007, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=16, network_alpha=8, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 16, "network_alpha": 8, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2200, 4000, 7000, 6),
            "expected_behavior": "顔再現を弱めに入れ、固定化を避ける候補。",
            "risk_note": "弱く出る場合はBalancedかStrongを検討してください。",
            "sort_order": 24,
        },
        {
            "id": "sdxl_character_face_adamw8bit_balanced",
            "name": "sdxl_character_face_adamw8bit_balanced",
            "display_name": "SDXL Character Face / AdamW8bit Balanced",
            "model_family": "SDXL",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "balanced",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=8, min_snr_gamma=5),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 8},
            "advanced_params": {"min_snr_gamma": 5},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "Character Face向けの安定した標準候補。",
            "risk_note": "まず比較基準として使います。",
            "sort_order": 26,
        },
        {
            "id": "sdxl_character_face_adamw8bit_strong",
            "name": "sdxl_character_face_adamw8bit_strong",
            "display_name": "SDXL Character Face / AdamW8bit Strong",
            "model_family": "SDXL",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "strong",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.00012, unet_lr=0.00012, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=48, network_alpha=24, train_batch_size=1, repeats=12, max_train_epochs=8),
            "basic_params": {"network_dim": 48, "network_alpha": 24, "train_batch_size": 1, "repeats": 12, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (3500, 6500, 9500, 6),
            "expected_behavior": "顔特徴を強めに出す比較候補。",
            "risk_note": "過学習、構図固定、style固定に注意してください。",
            "sort_order": 28,
        },
        {
            "id": "sdxl_character_face_paged_adamw8bit_balanced",
            "name": "sdxl_character_face_paged_adamw8bit_balanced",
            "display_name": "SDXL Character Face / PagedAdamW8bit Balanced",
            "model_family": "SDXL",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "PagedAdamW8bit",
            "optimizer_profile_id": "paged_adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "balanced",
            "params": sdxl_params(optimizer_type="PagedAdamW8bit", lr_scheduler="constant", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "AdamW8bitに近い省メモリ比較候補。",
            "risk_note": "環境依存で遅くなる場合があります。",
            "sort_order": 70,
        },
        {
            "id": "sdxl_character_face_lion_soft_experimental",
            "name": "sdxl_character_face_lion_soft_experimental",
            "display_name": "SDXL Character Face / Lion Soft / Experimental",
            "model_family": "SDXL",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "Lion",
            "optimizer_profile_id": "lion_sdxl_soft",
            "network_type_id": "standard_lora",
            "recipe_type": "experimental",
            "params": sdxl_params(optimizer_type="Lion", lr_scheduler="constant", learning_rate=0.00005, unet_lr=0.00005, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=24, network_alpha=12, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 24, "network_alpha": 12, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {"optimizer_args": {"weight_decay": 0.01}},
            "raw_args": {},
            "target": (2200, 3500, 7000, 6),
            "expected_behavior": "Lionを弱めに試す実験候補。",
            "risk_note": "Experimental。AdamWとの比較前提です。",
            "sort_order": 95,
        },
        {
            "id": "sdxl_style_adamw8bit_soft",
            "name": "sdxl_style_adamw8bit_soft",
            "display_name": "SDXL Style / AdamW8bit Soft",
            "model_family": "SDXL",
            "training_purpose_id": "style",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "soft",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.00007, unet_lr=0.00007, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=16, network_alpha=8, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 16, "network_alpha": 8, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2200, 4000, 7000, 6),
            "expected_behavior": "画風を弱めに乗せる候補。",
            "risk_note": "効果が弱い場合はBalancedを検討してください。",
            "sort_order": 210,
        },
        {
            "id": "sdxl_style_adamw8bit_balanced",
            "name": "sdxl_style_adamw8bit_balanced",
            "display_name": "SDXL Style / AdamW8bit Balanced",
            "model_family": "SDXL",
            "training_purpose_id": "style",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "balanced",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "Style LoRAの標準比較候補。",
            "risk_note": "対象キャラ固定化が出る場合は素材とcaptionを確認してください。",
            "sort_order": 220,
        },
        {
            "id": "sdxl_style_prodigy_soft_advanced",
            "name": "sdxl_style_prodigy_soft_advanced",
            "display_name": "SDXL Style / Prodigy Soft / Advanced",
            "model_family": "SDXL",
            "training_purpose_id": "style",
            "optimizer_definition_id": "Prodigy",
            "optimizer_profile_id": "prodigy_sdxl_soft",
            "network_type_id": "standard_lora",
            "recipe_type": "advanced",
            "params": sdxl_params(optimizer_type="Prodigy", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {"optimizer_args": {"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}},
            "raw_args": {},
            "target": (2200, 4000, 8000, 6),
            "expected_behavior": "Style向けにProdigy Softを比較する。",
            "risk_note": "画風固定が強く出る場合があります。",
            "sort_order": 230,
        },
        {
            "id": "sdxl_style_adafactor_auto_advanced",
            "name": "sdxl_style_adafactor_auto_advanced",
            "display_name": "SDXL Style / Adafactor Auto / Advanced",
            "model_family": "SDXL",
            "training_purpose_id": "style",
            "optimizer_definition_id": "Adafactor",
            "optimizer_profile_id": "adafactor_sdxl_auto",
            "network_type_id": "standard_lora",
            "recipe_type": "advanced",
            "params": sdxl_params(optimizer_type="Adafactor", lr_scheduler="adafactor", text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"relative_step": True, "scale_parameter": True, "warmup_init": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {"optimizer_args": {"relative_step": True, "scale_parameter": True, "warmup_init": True}},
            "raw_args": {},
            "target": (2200, 4000, 8000, 6),
            "expected_behavior": "Style向けにAdafactor Autoを比較する。",
            "risk_note": "通常LRとは意味が異なります。",
            "sort_order": 240,
        },
        {
            "id": "sdxl_style_lion_experimental",
            "name": "sdxl_style_lion_experimental",
            "display_name": "SDXL Style / Lion Experimental",
            "model_family": "SDXL",
            "training_purpose_id": "style",
            "optimizer_definition_id": "Lion",
            "optimizer_profile_id": "lion_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "experimental",
            "params": sdxl_params(optimizer_type="Lion", lr_scheduler="cosine", learning_rate=0.00005, unet_lr=0.00005, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"weight_decay": 0.01}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {"optimizer_args": {"weight_decay": 0.01}},
            "raw_args": {},
            "target": (2200, 4000, 8000, 6),
            "expected_behavior": "Style向けにLionを比較する。",
            "risk_note": "Experimental。比較用途として扱ってください。",
            "sort_order": 250,
        },
        {
            "id": "sdxl_costume_adamw8bit_balanced",
            "name": "sdxl_costume_adamw8bit_balanced",
            "display_name": "SDXL Costume / AdamW8bit Balanced",
            "model_family": "SDXL",
            "training_purpose_id": "costume",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "balanced",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2500, 5000, 8000, 6),
            "expected_behavior": "衣装再現の標準候補。",
            "risk_note": "キャラ顔への引っ張られ方を確認してください。",
            "sort_order": 310,
        },
        {
            "id": "sdxl_costume_adamw8bit_strong",
            "name": "sdxl_costume_adamw8bit_strong",
            "display_name": "SDXL Costume / AdamW8bit Strong",
            "model_family": "SDXL",
            "training_purpose_id": "costume",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sdxl_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "strong",
            "params": sdxl_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.00012, unet_lr=0.00012, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=48, network_alpha=24, train_batch_size=1, repeats=12, max_train_epochs=8),
            "basic_params": {"network_dim": 48, "network_alpha": 24, "train_batch_size": 1, "repeats": 12, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (3500, 6500, 9500, 6),
            "expected_behavior": "衣装特徴を強めに出す候補。",
            "risk_note": "汎用性が落ちる場合があります。",
            "sort_order": 320,
        },
        {
            "id": "sdxl_costume_prodigy_soft_advanced",
            "name": "sdxl_costume_prodigy_soft_advanced",
            "display_name": "SDXL Costume / Prodigy Soft / Advanced",
            "model_family": "SDXL",
            "training_purpose_id": "costume",
            "optimizer_definition_id": "Prodigy",
            "optimizer_profile_id": "prodigy_sdxl_soft",
            "network_type_id": "standard_lora",
            "recipe_type": "advanced",
            "params": sdxl_params(optimizer_type="Prodigy", lr_scheduler="constant", learning_rate=1.0, unet_lr=1.0, text_encoder_lr1=0, text_encoder_lr2=0, optimizer_args={"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=8, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 8, "max_train_epochs": 8},
            "advanced_params": {"optimizer_args": {"decouple": True, "weight_decay": 0.01, "d_coef": 0.5, "use_bias_correction": True}},
            "raw_args": {},
            "target": (2200, 4000, 8000, 6),
            "expected_behavior": "衣装向けにProdigy Softを比較する。",
            "risk_note": "強く効く場合があります。",
            "sort_order": 330,
        },
        {
            "id": "sd15_character_face_adamw8bit_balanced",
            "name": "sd15_character_face_adamw8bit_balanced",
            "display_name": "SD1.5 Character Face / AdamW8bit Balanced",
            "model_family": "SD15",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sd15_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "balanced",
            "params": sd15_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2000, 4000, 7000, 6),
            "expected_behavior": "SD1.5 Character Faceの標準候補。",
            "risk_note": "SDXLとはstep目安と解像度が異なります。",
            "sort_order": 410,
        },
        {
            "id": "sd15_character_face_adamw8bit_strong",
            "name": "sd15_character_face_adamw8bit_strong",
            "display_name": "SD1.5 Character Face / AdamW8bit Strong",
            "model_family": "SD15",
            "training_purpose_id": "character_face",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sd15_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "strong",
            "params": sd15_params(optimizer_type="AdamW8bit", lr_scheduler="constant", learning_rate=0.00012, unet_lr=0.00012, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=48, network_alpha=24, train_batch_size=1, repeats=12, max_train_epochs=8),
            "basic_params": {"network_dim": 48, "network_alpha": 24, "train_batch_size": 1, "repeats": 12, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (3000, 5500, 8500, 6),
            "expected_behavior": "SD1.5で顔特徴を強めに出す候補。",
            "risk_note": "過学習に注意してください。",
            "sort_order": 420,
        },
        {
            "id": "sd15_style_adamw8bit_balanced",
            "name": "sd15_style_adamw8bit_balanced",
            "display_name": "SD1.5 Style / AdamW8bit Balanced",
            "model_family": "SD15",
            "training_purpose_id": "style",
            "optimizer_definition_id": "AdamW8bit",
            "optimizer_profile_id": "adamw8bit_sd15_balanced",
            "network_type_id": "standard_lora",
            "recipe_type": "balanced",
            "params": sd15_params(optimizer_type="AdamW8bit", lr_scheduler="cosine", learning_rate=0.0001, unet_lr=0.0001, text_encoder_lr1=0, text_encoder_lr2=0, network_train_unet_only=True, cache_text_encoder_outputs=True, network_dim=32, network_alpha=16, train_batch_size=1, repeats=10, max_train_epochs=8),
            "basic_params": {"network_dim": 32, "network_alpha": 16, "train_batch_size": 1, "repeats": 10, "max_train_epochs": 8},
            "advanced_params": {},
            "raw_args": {},
            "target": (2000, 4000, 7000, 6),
            "expected_behavior": "SD1.5 Styleの標準候補。",
            "risk_note": "画風と素材構成の偏りを確認してください。",
            "sort_order": 430,
        },
    ]
)


def optimizer_definition_v2_rows(now: str):
    for item in OPTIMIZER_DEFINITIONS_V2:
        extra = optimizer_definition_extra(item)
        labels_json = json_dumps({"ja": item["display_name"], "en": item["display_name"]})
        descriptions_json = json_dumps({"ja": item["description"], "en": optimizer_description_en(item)})
        risk_notes_json = json_dumps({"ja": item["risk_note"], "en": optimizer_risk_note_en(item)})
        yield (
            item["id"],
            item["name"],
            item["display_name"],
            item["category"],
            item["lr_semantics"],
            item["default_learning_rate"],
            item["default_unet_lr"],
            item["default_text_encoder_lr"],
            item["default_scheduler"],
            json_dumps(item["allowed_schedulers"]),
            json_dumps(item["optimizer_args_schema"]),
            item["target_steps_min"],
            item["target_steps_recommended"],
            item["target_steps_max"],
            item["target_steps_confidence"],
            item["description"],
            item["risk_note"],
            json_dumps(item["compatibility_notes"]),
            extra["sd_scripts_optimizer_type"],
            json_dumps(extra["aliases"]),
            json_dumps(extra["required_dependencies"]),
            json_dumps(extra["required_params_schema"]),
            json_dumps(extra["recommended_params_schema"]),
            extra["lr_semantics_help"],
            extra["smoke_test_priority"],
            extra["validated_optimizer_type"],
            extra["validation_status"],
            labels_json,
            descriptions_json,
            risk_notes_json,
            1,
            1,
            now,
            now,
        )


def optimizer_profile_v2_rows(now: str):
    for row in OPTIMIZER_PROFILES_V2:
        bundle = optimizer_profile_param_bundle(row)
        (
            profile_id,
            optimizer_definition_id,
            model_family,
            profile_type,
            display_name,
            learning_rate,
            unet_lr,
            text_encoder_lr,
            scheduler,
            args,
            target_min,
            target_recommended,
            target_max,
            description,
            risk_note,
        ) = row
        labels_json = json_dumps({"ja": display_name, "en": display_name})
        descriptions_json = json_dumps(
            {"ja": description, "en": optimizer_profile_description_en(optimizer_definition_id, profile_type)}
        )
        risk_notes_json = json_dumps({"ja": risk_note, "en": optimizer_profile_risk_note_en(optimizer_definition_id)})
        yield (
            profile_id,
            optimizer_definition_id,
            model_family,
            profile_type,
            display_name,
            learning_rate,
            unet_lr,
            text_encoder_lr,
            scheduler,
            json_dumps(args),
            target_min,
            target_recommended,
            target_max,
            description,
            risk_note,
            bundle["sd_scripts_optimizer_type"],
            json_dumps(bundle["required_params"]),
            json_dumps(bundle["recommended_params"]),
            json_dumps(bundle["command_params"]),
            json_dumps(bundle["smoke_params"]),
            labels_json,
            descriptions_json,
            risk_notes_json,
            1,
            1,
            now,
            now,
        )


def network_type_rows(now: str):
    for row in NETWORK_TYPES:
        yield (*row[:6], json_dumps(row[6]), row[7], row[8], row[9], row[10], 1, 1, now, now)


def training_purpose_rows(now: str):
    for row in TRAINING_PURPOSES:
        purpose_id, name, display_name, description, roles, validation_preset, sample_template, overlay, sort_order = row
        yield (
            purpose_id,
            name,
            display_name,
            description,
            json_dumps(roles),
            validation_preset,
            sample_template,
            json_dumps(overlay),
            sort_order,
            1,
            1,
            now,
            now,
        )


PURPOSE_SHORT_LABELS = {
    "character_face": "顔キャラ",
    "character_full_body": "全身キャラ",
    "style": "画風",
    "costume": "衣装",
    "object": "物体",
    "concept": "概念",
    "custom": "Custom",
}

PURPOSE_SHORT_LABELS_EN = {
    "character_face": "Face Character",
    "character_full_body": "Full Body Character",
    "style": "Style",
    "costume": "Costume",
    "object": "Object",
    "concept": "Concept",
    "custom": "Custom",
}

RECIPE_TYPE_SHORT_LABELS = {
    "smoke": "動作確認",
    "pilot": "Pilot",
    "soft": "弱め汎用",
    "balanced": "標準",
    "balanced_long": "標準10epoch",
    "strong": "強め",
    "generalize": "汎化寄り",
    "advanced": "Advanced",
    "experimental": "実験",
    "custom": "Custom",
}

RECIPE_TYPE_SHORT_LABELS_EN = {
    "smoke": "Smoke",
    "pilot": "Pilot",
    "soft": "Soft",
    "balanced": "Balanced",
    "balanced_long": "Standard 10ep",
    "strong": "Strong",
    "generalize": "Generalize",
    "advanced": "Advanced",
    "experimental": "Experimental",
    "custom": "Custom",
}


def difficulty_label_for_recipe(recipe: dict[str, Any]) -> str:
    category = str(recipe.get("optimizer_definition_category") or "")
    recipe_type = str(recipe.get("recipe_type") or "")
    optimizer_id = str(recipe.get("optimizer_definition_id") or "")
    if "experimental" in category or recipe_type == "experimental" or optimizer_id in {"Lion", "Lion8bit", "PagedLion8bit", "DAdaptLion"}:
        return "experimental"
    if "advanced" in category or "auto_lr" in category or recipe_type == "advanced" or optimizer_id in {"Prodigy", "DAdaptAdam", "Adafactor"}:
        return "advanced"
    if recipe_type == "custom":
        return "custom"
    return "stable"


def recipe_display_labels(recipe: dict[str, Any]) -> dict[str, str]:
    purpose = PURPOSE_SHORT_LABELS.get(recipe.get("training_purpose_id"), recipe.get("training_purpose_id") or "Recipe")
    purpose_en = PURPOSE_SHORT_LABELS_EN.get(
        recipe.get("training_purpose_id"), str(recipe.get("training_purpose_id") or "Recipe").replace("_", " ").title()
    )
    recipe_type = str(recipe.get("recipe_type") or "")
    optimizer = recipe.get("optimizer_definition_id") or "Optimizer"
    network_type_id = str(recipe.get("network_type_id") or "standard_lora")
    network_label = "LoRA-C3Lier" if network_type_id == "lora_c3lier" else "Standard LoRA"
    params = recipe.get("params") or {}
    epochs = params.get("max_train_epochs") or recipe.get("max_train_epochs") or "-"
    dim = params.get("network_dim") or "-"
    network_args = params.get("network_args") if isinstance(params.get("network_args"), dict) else {}
    conv_dim = params.get("conv_dim") or network_args.get("conv_dim")
    lr = params.get("learning_rate") or params.get("unet_lr")
    target_min, target_recommended, target_max, _checkpoint_count = recipe["target"]
    type_label = RECIPE_TYPE_SHORT_LABELS.get(recipe_type, recipe_type or "Recipe")
    type_label_en = RECIPE_TYPE_SHORT_LABELS_EN.get(recipe_type, (recipe_type or "Recipe").replace("_", " ").title())

    display_name = str(recipe.get("display_name") or "")
    if optimizer == "Prodigy" and ("Soft" in display_name or recipe_type in {"soft", "advanced"}):
        short_label = f"{purpose}・Prodigy弱め"
        short_label_en = f"{purpose_en} - Prodigy Soft"
    elif optimizer == "Lion":
        short_label = f"{purpose}・Lion弱め実験" if "Soft" in display_name else f"{purpose}・Lion実験"
        short_label_en = f"{purpose_en} - Lion Soft Experimental" if "Soft" in display_name else f"{purpose_en} - Lion Experimental"
    elif optimizer == "Adafactor":
        short_label = f"{purpose}・Adafactor固定LR" if "Fixed" in display_name else f"{purpose}・Adafactor省メモリ"
        short_label_en = f"{purpose_en} - Adafactor Fixed LR" if "Fixed" in display_name else f"{purpose_en} - Adafactor Memory"
    elif optimizer.startswith("DAdapt"):
        short_label = f"{purpose}・DAdaptLion実験" if optimizer == "DAdaptLion" else f"{purpose}・DAdapt自動LR"
        short_label_en = f"{purpose_en} - DAdaptLion Experimental" if optimizer == "DAdaptLion" else f"{purpose_en} - DAdapt Auto LR"
    elif network_type_id == "lora_c3lier":
        short_label = f"{purpose}・C3Lier{type_label}"
        short_label_en = f"{purpose_en} - LoRA-C3Lier {type_label_en}"
    elif optimizer in {"Lion8bit", "PagedLion8bit"}:
        short_label = f"{purpose}・Lion実験"
        short_label_en = f"{purpose_en} - Lion Experimental"
    elif optimizer == "PagedAdamW8bit":
        short_label = f"{purpose}・省メモリ標準"
        short_label_en = f"{purpose_en} - Memory Balanced"
    else:
        short_label = f"{purpose}・{type_label}"
        short_label_en = f"{purpose_en} - {type_label_en}"

    subtitle_parts = [optimizer, network_label, f"{epochs}epoch", f"dim{dim}", f"{target_recommended}step目安"]
    subtitle_parts_en = [optimizer, network_label, f"{epochs} epochs", f"dim {dim}", f"target {target_recommended} steps"]
    if network_type_id == "lora_c3lier":
        subtitle_parts.extend([f"conv_dim {conv_dim or '-'}", "3x3 Conv拡張"])
        subtitle_parts_en.extend([f"conv_dim {conv_dim or '-'}", "3x3 Conv extension"])
    if recipe_type == "soft":
        subtitle_parts.append("LR低め")
        subtitle_parts_en.append("lower LR")
    if optimizer == "Prodigy":
        d_coef = (params.get("optimizer_args") or {}).get("d_coef") if isinstance(params.get("optimizer_args"), dict) else None
        subtitle_parts = ["Auto-LR", "Advanced", f"d_coef {d_coef or '-'}", "過学習注意"]
        subtitle_parts_en = ["Auto-LR", "Advanced", f"d_coef {d_coef or '-'}", "watch overtraining"]
    elif optimizer == "Lion":
        subtitle_parts = ["Lion", "Experimental", "LR低め", "比較用"]
        subtitle_parts_en = ["Lion", "Experimental", "lower LR", "comparison"]
    elif optimizer == "Adafactor":
        if "Fixed" in display_name:
            subtitle_parts = ["Adafactor", "Advanced", "固定LR", "max_grad_norm 0.0"]
            subtitle_parts_en = ["Adafactor", "Advanced", "fixed LR", "max_grad_norm 0.0"]
        else:
            subtitle_parts = ["Adafactor", "Advanced", "relative_step", "LR自動調整"]
            subtitle_parts_en = ["Adafactor", "Advanced", "relative_step", "auto LR"]
    elif optimizer.startswith("DAdapt"):
        subtitle_parts = ["Auto-LR", "Experimental" if optimizer == "DAdaptLion" else "Advanced", "倍率LR", "過学習注意"]
        subtitle_parts_en = ["Auto-LR", "Experimental" if optimizer == "DAdaptLion" else "Advanced", "LR multiplier", "watch overtraining"]
    elif recipe_type == "strong":
        subtitle_parts.append("効き強め")
        subtitle_parts_en.append("stronger effect")
    elif lr:
        subtitle_parts.append(f"LR {lr}")
        subtitle_parts_en.append(f"LR {lr}")

    full_label = f"[{recipe['model_family']}] {short_label} / {optimizer} / {epochs}epoch / {target_recommended}step目安"
    full_label_en = f"[{recipe['model_family']}] {short_label_en} / {optimizer} / {epochs} epochs / target {target_recommended} steps"
    direct_select_label = f"[{recipe['model_family']}] {short_label} / {optimizer} / {epochs}ep / dim{dim}"
    direct_select_label_en = f"[{recipe['model_family']}] {short_label_en} / {optimizer} / {epochs}ep / dim{dim}"
    difficulty = difficulty_label_for_recipe(recipe)
    recommended_badge = "おすすめ" if recipe_type in {"balanced", "balanced_long"} and difficulty == "stable" else ""
    return {
        "short_label": short_label,
        "short_label_en": short_label_en,
        "full_label": full_label,
        "full_label_en": full_label_en,
        "card_subtitle": " / ".join(str(part) for part in subtitle_parts if part not in {None, ""}),
        "card_subtitle_en": " / ".join(str(part) for part in subtitle_parts_en if part not in {None, ""}),
        "direct_select_label": direct_select_label,
        "direct_select_label_en": direct_select_label_en,
        "group_label": "",
        "recommended_badge": recommended_badge,
        "recommended_badge_en": "Recommended" if recommended_badge else "",
        "difficulty_label": difficulty,
    }


def recipe_expected_behavior_en(recipe: dict[str, Any]) -> str:
    purpose = PURPOSE_SHORT_LABELS_EN.get(
        recipe.get("training_purpose_id"), str(recipe.get("training_purpose_id") or "Recipe").replace("_", " ").title()
    )
    recipe_type = str(recipe.get("recipe_type") or "")
    optimizer = str(recipe.get("optimizer_definition_id") or "")
    if recipe_type == "smoke":
        return "Smoke Recipe for checking the execution path. Not intended for quality evaluation."
    if recipe_type == "pilot":
        return f"Lightweight pilot Recipe for checking loss trend and epoch differences for {purpose}."
    if recipe_type == "soft":
        return f"Lower-strength {purpose} Recipe designed to reduce fixation and overtraining risk."
    if recipe_type == "strong":
        return f"Stronger {purpose} Recipe for clearer LoRA effect. Compare epochs carefully."
    if optimizer == "Prodigy":
        return f"Auto-LR advanced {purpose} Recipe using Prodigy."
    if optimizer == "Adafactor":
        return f"Memory-oriented advanced {purpose} Recipe using Adafactor."
    if optimizer.startswith("DAdapt"):
        return f"Auto-LR advanced {purpose} Recipe using {optimizer}."
    if "Lion" in optimizer:
        return f"Experimental {purpose} Recipe using Lion for comparison."
    return f"Balanced {purpose} Recipe for standard LoRA training."


def recipe_risk_note_en(recipe: dict[str, Any]) -> str:
    recipe_type = str(recipe.get("recipe_type") or "")
    optimizer = str(recipe.get("optimizer_definition_id") or "")
    if recipe_type == "smoke":
        return "Do not use this for quality evaluation."
    if recipe_type == "pilot":
        return "Pilot output is not final quality. Use it only to inspect early trends."
    if recipe_type == "strong":
        return "Higher overtraining risk. Review candidate epochs and weights carefully."
    if optimizer == "Prodigy":
        return "Auto-LR can become strong. Watch overtraining and compare lower weights."
    if optimizer == "Adafactor":
        return "LR semantics differ by profile. Confirm relative_step or fixed LR before use."
    if optimizer.startswith("DAdapt"):
        return "Requires dadaptation and longer behavior checks. Keep Smoke/Mini Pilot results in mind."
    if "Lion" in optimizer:
        return "Experimental optimizer. Compare against an AdamW baseline."
    if optimizer == "PagedAdamW8bit":
        return "Memory-saving candidate. Confirm runtime support in the sd-scripts environment."
    return "Stable baseline. Still review candidate epochs before adopting a LoRA."


def training_recipe_v2_rows(now: str):
    for recipe in TRAINING_RECIPES_V2:
        target_min, target_recommended, target_max, checkpoint_count = recipe["target"]
        labels = recipe_display_labels(recipe)
        labels_json = json_dumps(
            {
                "ja": labels["short_label"],
                "en": labels["short_label_en"],
                "full_ja": labels["full_label"],
                "full_en": labels["full_label_en"],
                "direct_ja": labels["direct_select_label"],
                "direct_en": labels["direct_select_label_en"],
                "subtitle_ja": labels["card_subtitle"],
                "subtitle_en": labels["card_subtitle_en"],
                "badge_ja": labels["recommended_badge"],
                "badge_en": labels["recommended_badge_en"],
            }
        )
        descriptions_json = json_dumps({"ja": recipe["expected_behavior"], "en": recipe_expected_behavior_en(recipe)})
        risk_notes_json = json_dumps({"ja": recipe["risk_note"], "en": recipe_risk_note_en(recipe)})
        yield (
            recipe["id"],
            recipe["name"],
            recipe["display_name"],
            labels["short_label"],
            labels["full_label"],
            labels["card_subtitle"],
            labels["direct_select_label"],
            labels["group_label"],
            labels["recommended_badge"],
            labels["difficulty_label"],
            recipe["model_family"],
            recipe["training_purpose_id"],
            recipe["optimizer_definition_id"],
            recipe["optimizer_profile_id"],
            recipe["network_type_id"],
            recipe["recipe_type"],
            json_dumps(recipe["params"]),
            json_dumps(recipe["basic_params"]),
            json_dumps(recipe["advanced_params"]),
            json_dumps(recipe["raw_args"]),
            json_dumps({"phase": "12.1", "network_type_must_be_available": True}),
            target_min,
            target_recommended,
            target_max,
            checkpoint_count,
            recipe["expected_behavior"],
            recipe["risk_note"],
            recipe["sort_order"],
            1,
            1,
            labels_json,
            descriptions_json,
            risk_notes_json,
            now,
            now,
        )


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def merge_params(*parts: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in parts:
        if not part:
            continue
        for key, value in part.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                nested = dict(result[key])
                nested.update(value)
                result[key] = nested
            else:
                result[key] = value
    return result


def compatibility_check(params: dict[str, Any], *, network_type: dict[str, Any] | None = None, optimizer_definition: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    network_type = dict(network_type) if network_type else {}
    optimizer_definition = dict(optimizer_definition) if optimizer_definition else {}
    text_lr_values = [
        float(params.get(key) or 0)
        for key in ("text_encoder_lr", "text_encoder_lr1", "text_encoder_lr2")
        if params.get(key) not in (None, "")
    ]
    text_lr_active = any(value > 0 for value in text_lr_values)
    if params.get("cache_text_encoder_outputs") and text_lr_active:
        errors.append("cache_text_encoder_outputs=true かつ text_encoder_lr > 0 は併用できません。")
    if params.get("network_train_unet_only") and text_lr_active:
        errors.append("network_train_unet_only=true かつ text_encoder_lr > 0 は併用できません。")
    optimizer_type = str(params.get("optimizer_type") or optimizer_definition.get("id") or "")
    scheduler = str(params.get("lr_scheduler") or optimizer_definition.get("default_scheduler") or "")
    if optimizer_type in {"DAdaptAdam", "DAdaptLion", "Prodigy"}:
        if scheduler and scheduler != "constant":
            warnings.append(f"{optimizer_type} は lr_scheduler=constant を推奨します。")
        lr = params.get("learning_rate")
        try:
            lr_value = float(lr)
            if abs(lr_value - 1.0) > 1e-9:
                warnings.append(f"{optimizer_type} の learning_rate は通常LRではなく倍率です。1.0を基準にしてください。")
        except (TypeError, ValueError):
            pass
    if optimizer_type == "Adafactor":
        args = params.get("optimizer_args") if isinstance(params.get("optimizer_args"), dict) else {}
        relative_step = args.get("relative_step")
        if relative_step and params.get("learning_rate") not in (None, ""):
            warnings.append("Adafactor relative_step=Trueではlearning_rate指定の意味が通常と異なります。")
        if relative_step and scheduler != "adafactor":
            warnings.append("Adafactor Autoは lr_scheduler=adafactor を推奨します。")
        if relative_step is False:
            if scheduler != "constant_with_warmup":
                warnings.append("Adafactor Fixedは lr_scheduler=constant_with_warmup を推奨します。")
            try:
                if float(params.get("max_grad_norm") if params.get("max_grad_norm") is not None else 1.0) != 0.0:
                    warnings.append("Adafactor Fixedは max_grad_norm=0.0 を推奨します。")
            except (TypeError, ValueError):
                warnings.append("Adafactor Fixedは max_grad_norm=0.0 を推奨します。")
        if relative_step is None:
            warnings.append("Adafactorは optimizer_args.relative_step を明示してください。")
    if optimizer_type == "Lion":
        warnings.append("LionはExperimentalです。Smoke Test結果を確認してから使ってください。")
        try:
            if float(params.get("learning_rate") or 0) > 0.0001:
                warnings.append("Lionのlearning_rateは0.0001以下を推奨します。")
        except (TypeError, ValueError):
            pass
    if optimizer_type == "DAdaptLion":
        warnings.append("DAdaptLionはExperimentalです。DAdaptAdamより低優先の比較候補として扱います。")
    if optimizer_type in {"AdamW8bit", "PagedAdamW8bit", "Lion", "Lion8bit", "PagedLion8bit"}:
        try:
            if float(params.get("learning_rate") or 0) > 0.0002:
                warnings.append(f"{optimizer_type} の learning_rate が高めです。")
        except (TypeError, ValueError):
            pass
    if network_type.get("id") == "lora_c3lier":
        network_args = params.get("network_args") if isinstance(params.get("network_args"), dict) else {}
        conv_dim = params.get("conv_dim", network_args.get("conv_dim"))
        conv_alpha = params.get("conv_alpha", network_args.get("conv_alpha"))
        conv_dim_value: int | None = None
        conv_alpha_value: int | None = None
        try:
            conv_dim_value = int(conv_dim)
            if conv_dim_value <= 0:
                errors.append("LoRA-C3Lierでは conv_dim > 0 が必要です。")
        except (TypeError, ValueError):
            errors.append("LoRA-C3Lierでは conv_dim > 0 が必要です。")
        try:
            conv_alpha_value = int(conv_alpha)
            if conv_alpha_value <= 0:
                errors.append("LoRA-C3Lierでは conv_alpha > 0 が必要です。")
        except (TypeError, ValueError):
            errors.append("LoRA-C3Lierでは conv_alpha > 0 が必要です。")
        if conv_dim_value is not None and conv_alpha_value is not None and conv_alpha_value > conv_dim_value:
            warnings.append("LoRA-C3Lierの conv_alpha が conv_dim より大きいです。まず conv_alpha <= conv_dim を推奨します。")
    availability = network_type.get("availability")
    if availability and availability != "available":
        errors.append(f"Network type {network_type.get('display_name') or network_type.get('id')} は {availability} のため通常実行できません。")
    return {"errors": errors, "warnings": warnings, "ok": not errors}


def recipe_v2_snapshot(recipe: Any, optimizer_definition: Any, optimizer_profile: Any, network_type: Any, purpose: Any) -> dict[str, Any]:
    return {
        "recipe": dict(recipe) if recipe else None,
        "optimizer_definition": dict(optimizer_definition) if optimizer_definition else None,
        "optimizer_profile": dict(optimizer_profile) if optimizer_profile else None,
        "network_type": dict(network_type) if network_type else None,
        "training_purpose": dict(purpose) if purpose else None,
    }
